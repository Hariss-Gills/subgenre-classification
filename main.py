import ast
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def parse_list(list_str):
    return ast.literal_eval(list_str)


def main():
    model_results = pd.read_csv("results/model_results.csv")

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


if __name__ == "__main__":
    main()
