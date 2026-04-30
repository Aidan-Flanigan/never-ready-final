"""
Bar-chart helper. Called by run_all.py.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    "CatBoost":                          "CatBoost",
}

COLORS = ["#2E86AB", "#E84855", "#3BB273", "#F4A261"]


def plot_metric_bar_chart(results_csv, output_path, target_name=None):
    """
    Save a grouped bar chart of precision / recall / F1 / accuracy from a results CSV.
    """
    
    df = pd.read_csv(results_csv)
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

    title_target = target_name or os.path.splitext(os.path.basename(results_csv))[0]
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Model Comparison \u2014 {title_target}",
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
    print(f"Bar chart saved to {output_path}")
