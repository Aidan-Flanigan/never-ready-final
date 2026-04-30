"""
ROC-curve helper. Called by run_all.py with already-fitted models.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_curve, auc

COLORS = [
    "#2E86AB", "#E84855", "#3BB273",
    "#F4A261", "#9B5DE5", "#00BBF9",
    "#FF7F50",
]


def plot_roc_curves(
    fitted_models,
    X_test,
    y_test,
    available_nb_vars,
    comparison,
    target_name,
    output_path,
):
    """
    Save one ROC-curve panel covering every fitted model that supports predict_proba.
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for (model_name, model), color in zip(fitted_models.items(), COLORS):
        if not hasattr(model, "predict_proba"):
            continue

        if model_name == "Bernoulli Naive Bayes" and available_nb_vars:
            X_test_use = X_test[available_nb_vars]
        else:
            X_test_use = X_test

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
            color=color, linewidth=2,
            label=f"{model_name}  (AUC = {roc_auc:.3f})",
        )
        ax.scatter(
            fpr[threshold_idx], tpr[threshold_idx],
            color=color, s=80, zorder=5, edgecolors="white", linewidths=1.2,
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="#AAAAAA", linewidth=1.2,
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax.set_title(
        f"ROC Curves by Model \u2014 {target_name}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    dot_note = mlines.Line2D(
        [], [], color="gray", marker="o", linestyle="None",
        markersize=7, markeredgecolor="white",
        label="\u25cf = selected threshold",
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [dot_note],
        fontsize=9, framealpha=0.6, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved to {output_path}")
