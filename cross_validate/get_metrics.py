import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.utils import set_random_seed

RANDOM_STATE = 100


def load_and_preprocess_data(filepath="../results/audio_features.csv"):
    """Load and preprocess dataset."""
    dataset = pd.read_csv(filepath).drop(columns=["Unnamed: 0", "Track ID", "Slice"])
    X = dataset.drop(columns=["Subgenre"])
    y = dataset["Subgenre"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder


def plot_confusion_matrix(confusions, class_labels, model_name):
    """Plot and save the confusion matrix."""
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
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(y_test, y_pred):
    """Evaluate a model, returning metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, f1_weighted, confusion


def build_deep_nn(input_shape, num_classes):
    """Build a deep neural network."""
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


def evaluate_deep_nn(X, y, label_encoder):
    """Evaluate a deep neural network using k-fold cross-validation."""
    set_random_seed(RANDOM_STATE)
    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
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

    plot_confusion_matrix(confusions, label_encoder.classes_, "Deep NN")
    return [{"Model": "Deep NN", "Accuracies": accuracies, "F1 Scores": f1_scores}]


def evaluate_non_deep_models(X, y, label_encoder):
    """Evaluate multiple non-deep models using k-fold cross-validation."""
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
            hidden_layer_sizes=(5000, 10), max_iter=2000, random_state=RANDOM_STATE
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

        plot_confusion_matrix(confusions, label_encoder.classes_, name)
        results.append(
            {"Model": name, "Accuracies": accuracies, "F1 Scores": f1_scores}
        )

    return results


def main():
    X, y, label_encoder = load_and_preprocess_data()

    deep_nn_results = evaluate_deep_nn(X, y, label_encoder)
    non_deep_results = evaluate_non_deep_models(X, y, label_encoder)

    results_df = pd.DataFrame(deep_nn_results + non_deep_results)
    results_df.to_csv("../results/model_results.csv", index=False)


if __name__ == "__main__":
    main()
