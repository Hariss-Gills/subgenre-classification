import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd
from typing import Any
import numpy as np
import os
import librosa
import requests
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from keras.utils import set_random_seed

load_dotenv()
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.environ["CLIENT_ID_ZK"],
        client_secret=os.environ["CLIENT_SECRET_ZK"],
    )
)

METAL_SUBGENRES = [
    "Black",
    "Death",
    "Doom",
    "Sludge",
    "Industrial",
    "Experimental",
    "Folk",
    "Gothic",
    "Grindcore",
    "Groove",
    "Heavy",
    "Metalcore",
    "Deathcore",
    "Power",
    "Progressive",
    "Speed",
    "Symphonic",
    "Thrash",
]
FEATURE_COLUMNS = [
    "Track ID",
    "Subgenre",
    "Slice",
    "Chroma",
    "RMS",
    "Spectral Centroid",
    "Spectral Bandwidth",
    "Spectral Rolloff",
    "Zero Crossing Rate",
    "MFCC",
    "Harmony",
    "Tempo",
]
MAX_PLAYLISTS = 5
MIN_TRACKS_IN_PLAYLIST = 100
TOP_TRACKS_LIMIT = 100
SLICE_DURATION = 3
RANDOM_STATE = 100

# logger = logging.getLogger("spotipy")
# logger.setLevel(logging.DEBUG)

# Create console handler to output log messages
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# Create a formatter for log messages
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# ch.setFormatter(formatter)


# Add the handler to the logger
# logger.addHandler(ch)


def mean_plus_variance(feature: np.ndarray) -> float:
    """
    Calculate the sum of the mean and the variance (square of the standard deviation)
    of a numeric feature array.

    Args:
        feature np.ndarray: A numpy array object containing numeric values.

    Returns:
        float: The calculated value (mean + variance).
    """
    return np.mean(feature) + np.std(feature) ** 2


def extract_features(slice_data: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Extract audio features from a given slice of audio data.

    Args:
        slice_data (np.ndarray): Audio slice data.
        sr (int): Sampling rate.

    Returns:
        Dict[str, Any]: A dictionary of computed audio features.
    """
    return {
        "Chroma": mean_plus_variance(librosa.feature.chroma_stft(y=slice_data, sr=sr)),
        "RMS": mean_plus_variance(librosa.feature.rms(y=slice_data)),
        "Spectral Centroid": mean_plus_variance(
            librosa.feature.spectral_centroid(y=slice_data, sr=sr)
        ),
        "Spectral Bandwidth": mean_plus_variance(
            librosa.feature.spectral_bandwidth(y=slice_data, sr=sr)
        ),
        "Spectral Rolloff": mean_plus_variance(
            librosa.feature.spectral_rolloff(y=slice_data, sr=sr)
        ),
        "Zero Crossing Rate": mean_plus_variance(
            librosa.feature.zero_crossing_rate(y=slice_data)
        ),
        "MFCC": mean_plus_variance(
            librosa.feature.mfcc(y=slice_data, sr=sr, n_mfcc=20)
        ),
        "Harmony": mean_plus_variance(librosa.effects.harmonic(y=slice_data)),
        "Tempo": librosa.beat.tempo(y=slice_data, sr=sr).mean(),
    }


def get_top_100_tracks_for_all_subgenres(subgenres: list[str]) -> pd.DataFrame:
    """
    Retrieves the top 100 tracks for each given subgenre from Spotify playlists.
    Since we are processing lots of data, if an error occurs we return the DataFrame anyway.

    Args:
        subgenres (List[str]): A list of subgenres to search for.

    Returns:
        pd.DataFrame: A DataFrame containing track IDs and their associated subgenres.
    """
    labelled_tracks = []
    try:
        for subgenre in subgenres:
            playlist_results = sp.search(
                q=f"{subgenre} Metal", type="playlist", limit=MAX_PLAYLISTS
            )

            selected_playlist_id = next(
                (
                    playlist["id"]
                    for playlist in playlist_results["playlists"]["items"]
                    if playlist["tracks"]["total"] >= MIN_TRACKS_IN_PLAYLIST
                ),
                None,
            )

            playlist_tracks = sp.playlist_tracks(
                selected_playlist_id, limit=TOP_TRACKS_LIMIT
            )
            track_list = [
                track["track"]
                for track in playlist_tracks["items"]
                if track["track"] is not None
            ]

            sorted_tracks = sorted(
                track_list, key=lambda x: x["popularity"], reverse=True
            )
            labelled_tracks.extend(
                [
                    {"Track ID": track["id"], "Subgenre": subgenre}
                    for track in sorted_tracks
                ]
            )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        return pd.DataFrame(labelled_tracks, columns=["Track ID", "Subgenre"])


def extract_audio_features_from_preview(
    df: pd.DataFrame, start_index: int = 0
) -> pd.DataFrame:
    """
    Extract audio features (MEAN + SD²) for 3-second slices of preview URLs.
    Since we are processing lots of data, if an error occurs we return the DataFrame anyway.
    Args:
        df (pd.DataFrame): DataFrame with columns "Track ID" and "Subgenre".
        start_index (int): Index to start processing from (default is 0).

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    feature_results = []

    try:
        for index, row in df.iloc[start_index:].iterrows():
            print(f"Processing index: {index:04d}", end="\r")
            track_id = row["Track ID"]
            subgenre = row["Subgenre"]

            track_details = sp.track(track_id)
            preview_url = track_details.get("preview_url")

            if not preview_url:
                print(f"No preview URL for track {track_id}")
                continue

            response = requests.get(preview_url, timeout=30)
            y, sr = librosa.load(BytesIO(response.content), sr=None)

            # Always returns 29.71265306122449 hence last slice is ignored
            duration = librosa.get_duration(y=y, sr=sr)
            num_slices = int(duration // SLICE_DURATION)
            for i in range(num_slices):
                start_sample = i * SLICE_DURATION * sr
                end_sample = start_sample + SLICE_DURATION * sr
                slice_data = y[int(start_sample) : int(end_sample)]
                features = extract_features(slice_data, sr)
                features.update(
                    {"Track ID": track_id, "Subgenre": subgenre, "Slice": i + 1}
                )
                feature_results.append(features)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        return pd.DataFrame(feature_results, columns=FEATURE_COLUMNS)


def create_distribution_plot(audio_dataset):
    subgenre_counts = audio_dataset["Subgenre"].value_counts()
    plt.figure(figsize=(12, 6))
    subgenre_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Distribution of Subgenres", fontsize=12)
    plt.xlabel("Subgenre", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.savefig("subgenre_distribution.png")


def build_deep_nn(input_shape, num_classes):
    """
    Build a deep neural network for music genre classification.

    Args:
        input_shape (tuple): Shape of the input features (number of features, ).
        num_classes (int): Number of output classes.

    Returns:
        Model: Compiled neural network model.
    """
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def evaluate_nn_with_kfold():
    """
    Evaluate a deep neural network using repeated stratified k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels (encoded).
        label_encoder (LabelEncoder): Encoder for target labels.
        num_folds (int): Number of folds for k-fold cross-validation.
        num_repeats (int): Number of repetitions for repeated stratified k-fold.

    Returns:
        dict: Dictionary containing average accuracy and standard deviation.
    """
    set_random_seed(RANDOM_STATE)
    features_df = pd.read_csv("audio_features.csv").drop(
        columns=["Unnamed: 0", "Track ID", "Slice"]
    )

    results_df = pd.read_csv("model_results.csv")

    X = features_df.drop(columns=["Subgenre"])
    y = features_df["Subgenre"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X = features_df.drop(
        columns=[
            "Subgenre",
        ]
    )

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
    accuracies = []
    f1_scores = []
    confusions = []
    results = []
    test = []
    for train_index, test_index in rkf.split(X, y):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert targets to categorical (one-hot encoding)
        y_train_cat = to_categorical(y_train, num_classes=len(label_encoder.classes_))
        y_test_cat = to_categorical(y_test, num_classes=len(label_encoder.classes_))

        # Build the model
        model = build_deep_nn(
            input_shape=(X_train.shape[1],), num_classes=len(label_encoder.classes_)
        )

        # Train the model
        model.fit(
            X_train,
            y_train_cat,
            epochs=20,
            batch_size=32,
            verbose=0,  # Suppress training output for clarity
        )

        _, accuracy_keras = model.evaluate(X_test, y_test_cat, verbose=0)

        # Evaluate the model
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        confusion = confusion_matrix(y_test, y_pred)

        test.append(accuracy_keras)
        accuracies.append(accuracy)
        f1_scores.append(f1_weighted)
        confusions.append(confusion)
        accuracies.append(accuracy)

    avg_confusion = np.mean(confusions, axis=0)
    sum_conf = np.sum(confusions, axis=0)
    # Plot confusion matrix with improved formatting
    plt.figure(figsize=(20, 16))  # Increase figure size significantly
    disp = ConfusionMatrixDisplay(
        avg_confusion.round().astype(int), display_labels=label_encoder.classes_
    )

    # Customize the plot for better readability
    disp.plot(
        cmap="Blues",
        values_format="d",  # Ensure integer display
        text_kw={"fontsize": 8},  # Adjust font size for many labels
    )

    plt.title(f"Confusion Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate and align x-axis labels
    plt.yticks(rotation=45, fontsize=8)  # Rotate and align y-axis labels

    plt.tight_layout(pad=3.0)  # Add extra padding
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    # Save the confusion matrix as an imageplt.
    plt.savefig(f"confusion_matrix_deep.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot after saving to avoid overlap in future plots

    results.append(
        {
            "Model": "Deep",
            "Accuracies": accuracies,
            "F1 Scores": f1_scores,
        }
    )
    new_row = pd.DataFrame(results)
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv("model_results_deep_less_feat.csv", index=False)


def evaluate_model(y_test, y_pred):
    """Evaluate a model, returning metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, f1_weighted, confusion


def test_models_2():
    # Load and preprocess dataset
    audio_dataset = pd.read_csv("audio_features.csv").drop(
        columns=["Unnamed: 0", "Track ID", "Slice"]
    )

    # Separate features and target
    X = audio_dataset.drop(
        columns=[
            "Subgenre",
        ]
    )
    y = audio_dataset["Subgenre"]

    # Encode the target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Encoded labels:", y_encoded)
    print("Classes:", label_encoder.classes_)

    # Cross-validation strategy
    repeated_kfold = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, penalty="l2"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            min_impurity_decrease=np.exp(-5),
            random_state=RANDOM_STATE,
        ),
        "k-NN": KNeighborsClassifier(n_neighbors=1),
        "Naïve Bayes": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(5000, 10),
            activation="relu",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }

    # Collect results
    results = []

    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        accuracies = []
        f1_scores = []
        confusions = []

        for train_index, test_index in repeated_kfold.split(X, y_encoded):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy, f1_weighted, confusion = evaluate_model(y_test, y_pred)
            accuracies.append(accuracy)
            f1_scores.append(f1_weighted)
            confusions.append(confusion)

        avg_confusion = np.mean(confusions, axis=0)
        sum_conf = np.sum(confusions, axis=0)
        # Plot confusion matrix with improved formatting
        plt.figure(figsize=(20, 16))  # Increase figure size significantly
        disp = ConfusionMatrixDisplay(
            avg_confusion.round().astype(int), display_labels=label_encoder.classes_
        )

        # Customize the plot for better readability
        disp.plot(
            cmap="Blues",
            values_format="d",  # Ensure integer display
            text_kw={"fontsize": 8},  # Adjust font size for many labels
        )

        plt.title(f"Confusion Matrix", fontsize=16)
        plt.xticks(
            rotation=45, ha="right", fontsize=8
        )  # Rotate and align x-axis labels
        plt.yticks(rotation=45, fontsize=8)  # Rotate and align y-axis labels

        plt.tight_layout(pad=3.0)  # Add extra padding
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        # Save the confusion matrix as an imageplt.
        plt.savefig(f"confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot after saving to avoid overlap in future plots

        results.append(
            {
                "Model": name,
                "Accuracies": accuracies,
                "F1 Scores": f1_scores,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)


def plot_feature_differences_per_label(features_df):
    """
    Plots the differences in audio features for each subgenre.

    Args:
        features_df (pd.DataFrame): DataFrame containing audio features and their associated subgenres.
    """
    # Extract numeric features and group by subgenre
    feature_columns = features_df.columns.drop("Subgenre")
    grouped = features_df.groupby("Subgenre")[feature_columns]

    # Calculate mean and standard deviation for each feature per subgenre
    mean_features = grouped.mean()
    std_features = grouped.std()

    # Create subplots for each feature
    plt.figure(figsize=(20, len(feature_columns) * 4))
    for i, feature in enumerate(feature_columns, start=1):
        plt.subplot(len(feature_columns), 1, i)
        plt.errorbar(
            mean_features.index,
            mean_features[feature],
            yerr=std_features[feature],
            fmt="o-",
            label=feature,
            capsize=5,
        )
        plt.title(f"Feature: {feature}", fontsize=14)
        plt.xlabel("Subgenre", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("feature_differences_per_label.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # test_models_2()
    # evaluate_nn_with_kfold()
    import numpy as np
    from scipy import stats
    import ast
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    model_results = pd.read_csv("model_results.csv")

    # Function to convert string representation of list to actual list
    def parse_list(list_str):
        return ast.literal_eval(list_str)

    # Read the accuracies for each model
    logistic_accuracies = parse_list(
        model_results.loc[
            model_results["Model"] == "Logistic Regression", "Accuracies"
        ].values[0]
    )
    random_forest_accuracies = parse_list(
        model_results.loc[
            model_results["Model"] == "Random Forest", "Accuracies"
        ].values[0]
    )
    knn_accuracies = parse_list(
        model_results.loc[model_results["Model"] == "k-NN", "Accuracies"].values[0]
    )
    naive_bayes_accuracies = parse_list(
        model_results.loc[model_results["Model"] == "Naïve Bayes", "Accuracies"].values[
            0
        ]
    )
    mlp_accuracies = parse_list(
        model_results.loc[model_results["Model"] == "MLP", "Accuracies"].values[0]
    )
    deep_accuracies = parse_list(
        model_results.loc[model_results["Model"] == "Deep", "Accuracies"].values[0]
    )

    # Perform Shapiro-Wilk test for each model
    models = {
        "Logistic Regression": logistic_accuracies,
        "Random Forest": random_forest_accuracies,
        "k-NN": knn_accuracies,
        "Naïve Bayes": naive_bayes_accuracies,
        "MLP": mlp_accuracies,
        "Deep": deep_accuracies,
    }

    print("Shapiro-Wilk Test Results:")
    for model_name, accuracies in models.items():
        statistic, p_value = stats.shapiro(accuracies)
        print(f"{model_name}:")
        print(f"  Statistic: {statistic}")
        print(f"  p-value: {p_value}")
        print(f"  Normally distributed: {p_value > 0.05}\n")

    f_statistic, p_value = stats.f_oneway(
        logistic_accuracies,
        random_forest_accuracies,
        knn_accuracies,
        naive_bayes_accuracies,
        mlp_accuracies,
        deep_accuracies,
    )

    print("ANOVA Test Results:")
    print(f"F-statistic: {f_statistic}")
    print(f"p-value: {p_value}")

    # Interpret the results
    if p_value < 0.05:
        print("There is a significant difference between the models' accuracies.")
        accuracies = (
            logistic_accuracies
            + random_forest_accuracies
            + knn_accuracies
            + naive_bayes_accuracies
            + mlp_accuracies
            + deep_accuracies
        )
        models_repeated = (
            ["Logistic Regression"] * len(logistic_accuracies)
            + ["Random Forest"] * len(random_forest_accuracies)
            + ["k-NN"] * len(knn_accuracies)
            + ["Naïve Bayes"] * len(naive_bayes_accuracies)
            + ["MLP"] * len(mlp_accuracies)
            + ["Deep"] * len(deep_accuracies)
        )

        df = pd.DataFrame({"Accuracy": accuracies, "Model": models_repeated})

        tukey_result = pairwise_tukeyhsd(
            endog=df["Accuracy"], groups=df["Model"], alpha=0.05
        )
        print("\nTukey HSD Test Results:")
        print(tukey_result)
    else:
        print("There is no significant difference between the models' accuracies.")
