import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import set_random_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 100


def load_and_preprocess_data(
    filepath: str = "./results/audio_features.csv",
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    LabelEncoder,
]:
    """
    Load and preprocess dataset from a given CSV file.

    Args:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A tuple containing:
            - A DataFrame with features (X).
            - An array of encoded labels (y).
            - The label encoder used to transform the labels.
    """
    dataset = pd.read_csv(filepath).drop(
        columns=["Unnamed: 0", "Track ID", "Slice"]
    )
    X = dataset.drop(columns=["Subgenre"])
    y = dataset["Subgenre"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, np.array(y_encoded), label_encoder


def plot_confusion_matrix(
    confusions: list[np.ndarray],
    class_labels: np.ndarray,
    model_name: str,
) -> None:
    """
    Plot and save the confusion matrix based on average confusion across folds.

    Args:
        confusions (list of np.ndarray): List of confusion matrices
        from different folds.
        class_labels (np.ndarray): Array of class labels.
        model_name (str): The name of the model to be displayed
        in the plot title.
    """
    avg_confusion = np.mean(confusions, axis=0)
    plt.figure(figsize=(20, 16))
    disp = ConfusionMatrixDisplay(
        avg_confusion.round().astype(int), display_labels=class_labels
    )
    disp.plot(cmap="Blues", values_format="d", text_kw={"fontsize": 8})
    plt.title(f"Confusion Matrix: {model_name}", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=45, fontsize=8)
    plt.tight_layout(pad=3.0)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.savefig(
        f"./results/confusion_matrix_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def evaluate_model(
    y_test: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate a model using accuracy, F1 score, and confusion matrix.

    Args:
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        tuple: A tuple containing:
            - Accuracy score.
            - Weighted F1 score.
            - Confusion matrix.
    """
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, f1_weighted, confusion


def build_deep_nn(input_shape: tuple[int], num_classes: int) -> Sequential:
    """
    Build a deep neural network model.

    Args:
        input_shape (tuple[int]): The shape of the input data.
        num_classes (int): The number of output classes.

    Returns:
        Sequential: A Keras Sequential model instance.
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


def evaluate_deep_nn(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> list[dict]:
    """
    Evaluate a deep neural network using k-fold cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Label array.
        label_encoder (LabelEncoder): The label encoder used for
        transforming labels.

    Returns:
        list of dict: List containing results for the deep
        neural network model.
    """
    set_random_seed(RANDOM_STATE)
    rkf = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )
    num_classes = 0
    if label_encoder.classes_ is not None:
        num_classes = len(label_encoder.classes_)
    accuracies, f1_scores, confusions = [], [], []

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_cat = to_categorical(y_train, num_classes)

        model = build_deep_nn((X_train.shape[1],), num_classes)
        model.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=0)

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy, f1_weighted, confusion = evaluate_model(y_test, y_pred)
        accuracies.append(accuracy)
        f1_scores.append(f1_weighted)
        confusions.append(confusion)

    plot_confusion_matrix(
        confusions, np.array(label_encoder.classes_), "Deep NN"
    )
    return [
        {"Model": "Deep NN", "Accuracies": accuracies, "F1 Scores": f1_scores}
    ]


def evaluate_non_deep_models(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> list[dict]:
    """
    Evaluate multiple non-deep models using k-fold cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Label array.
        label_encoder (LabelEncoder): The label encoder used for
        transforming labels.

    Returns:
        list of dict: List containing evaluation results
        for each non-deep model.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=1000, max_depth=10, random_state=RANDOM_STATE
        ),
        "k-NN": KNeighborsClassifier(n_neighbors=1),
        "Naïve Bayes": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(5000, 10),
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }
    repeated_kfold = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )
    results = []

    for name, model in models.items():
        accuracies, f1_scores, confusions = [], [], []

        for train_idx, test_idx in repeated_kfold.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy, f1_weighted, confusion = evaluate_model(y_test, y_pred)
            accuracies.append(accuracy)
            f1_scores.append(f1_weighted)
            confusions.append(confusion)

        plot_confusion_matrix(
            confusions, np.array(label_encoder.classes_), name
        )
        results.append(
            {"Model": name, "Accuracies": accuracies, "F1 Scores": f1_scores}
        )

    return results


def main() -> None:
    """
    Main function to load data, evaluate models, and save results.

    This function loads the dataset, evaluates both deep and non-deep models,
    and saves the evaluation results to a CSV file.
    """
    X, y, label_encoder = load_and_preprocess_data(
        "./results/audio_features.csv"
    )

    deep_nn_results = evaluate_deep_nn(X, y, label_encoder)
    non_deep_results = evaluate_non_deep_models(X, y, label_encoder)

    results_df = pd.DataFrame(deep_nn_results + non_deep_results)
    results_df.to_csv("./results/model_results_1.csv", index=False)


if __name__ == "__main__":
    main()
