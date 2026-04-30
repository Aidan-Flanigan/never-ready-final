import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from targets import TARGETS

PLOTS_DIR = "visuals/plots"
METRICS = ["precision", "recall", "f1", "accuracy"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "accuracy": "Accuracy",
}

SHORT_NAMES = {
    "Baseline Logistic Regression":      "Logistic\nRegression",
    "LASSO Logistic Regression with CV": "LASSO\nLogistic CV",
    "Decision Tree":                     "Decision\nTree",
    "Random Forest":                     "Random\nForest",
    "Bernoulli Naive Bayes":             "Naive\nBayes",
    "Gradient Boosting":                 "Gradient\nBoosting",
}

COLORS = ["#2E86AB", "#E84855", "#3BB273", "#F4A261"]


def plot_bar_chart(csv_path, output_path, title_suffix):
    df = pd.read_csv(csv_path)
    df["model_label"] = df["model"].map(SHORT_NAMES).fillna(df["model"])

    models = df["model_label"].tolist()
    n_models = len(models)
    n_metrics = len(METRICS)

    x = np.arange(n_models)
    bar_width = 0.22
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, color) in enumerate(zip(METRICS, COLORS)):
        values = df[metric].tolist()
        bars = ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            label=METRIC_LABELS[metric],
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Model Comparison ({title_suffix}) — Accuracy, Precision, Recall, F1",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.5)

    if "threshold" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            if pd.notna(row["threshold"]):
                ax.text(
                    x[i], -0.09,
                    f"t={row['threshold']:.3f}",
                    ha="center", va="top",
                    fontsize=7.5, color="#666666",
                    transform=ax.get_xaxis_transform(),
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {output_path}")


def main():
    for target in TARGETS:
        name = target["name"]
        csv_path = target["results_csv"]
        output_path = os.path.join(PLOTS_DIR, f"{name}_comparison.png")

        if not os.path.exists(csv_path):
            print(f"Skipping {name}: {csv_path} not found")
            continue

        plot_bar_chart(csv_path, output_path, title_suffix=name)


if __name__ == "__main__":
    main()
