import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

sys.path.append("model-comparison")
from functions import run_classification_models  # noqa: E402

from targets import TARGETS, KNOWLEDGE_VARS, NB_VARS, RAW_MISSING_MAP

DATA_PATH = "data/data.csv"
PLOTS_DIR = "visuals/plots"

COLORS = [
    "#2E86AB", "#E84855", "#3BB273",
    "#F4A261", "#9B5DE5", "#00BBF9",
]


def plot_roc_for_target(target, raw_df):
    name = target["name"]
    target_col = target["target_col"]
    output_path = os.path.join(PLOTS_DIR, f"{name}_roc_auc.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = target["prep"](raw_df)

    comparison, fitted_models = run_classification_models(
        df=df,
        target_col=target_col,
        predictor_vars=KNOWLEDGE_VARS,
        nb_vars=NB_VARS,
        raw_missing_map=RAW_MISSING_MAP,
        target_name=name,
        test_size=0.25,
        random_state=42,
        tune_threshold=True,
        threshold_metric=target["threshold_metric"],
        results_csv=target["results_csv"],
    )

    df_model = df[df[target_col].isin([0, 1])].copy()
    X = df_model[KNOWLEDGE_VARS].copy()
    for col, missing_vals in RAW_MISSING_MAP.items():
        if col in X.columns:
            X[col] = X[col].replace(missing_vals, np.nan)
    y = df_model[target_col]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    for (model_name, model), color in zip(fitted_models.items(), COLORS):
        if not hasattr(model, "predict_proba"):
            continue

        X_test_use = X_test[NB_VARS] if model_name == "Bernoulli Naive Bayes" else X_test

        y_prob = model.predict_proba(X_test_use)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        threshold_match = comparison.loc[
            comparison["model"] == model_name, "threshold"
        ].values
        model_threshold = float(threshold_match[0]) if len(threshold_match) > 0 else 0.5

        threshold_idx = int(np.argmin(np.abs(thresholds - model_threshold)))

        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f"{model_name}  (AUC = {roc_auc:.3f})",
        )
        ax.scatter(
            fpr[threshold_idx], tpr[threshold_idx],
            color=color, s=80, zorder=5, edgecolors="white", linewidths=1.2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", linewidth=1.2,
            label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax.set_title(f"ROC Curves by Model — {name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    dot_note = mlines.Line2D(
        [], [], color="gray", marker="o", linestyle="None",
        markersize=7, markeredgecolor="white",
        label="● = selected threshold",
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [dot_note],
        fontsize=9, framealpha=0.6,
        loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved to {output_path}")


def main():
    raw_df = pd.read_csv(DATA_PATH)
    for target in TARGETS:
        print(f"=== {target['name']} ===")
        plot_roc_for_target(target, raw_df)


if __name__ == "__main__":
    main()
