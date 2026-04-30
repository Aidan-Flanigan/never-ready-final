"""
Confusion-matrix helper. Called by run_all.py with already-fitted models.
"""
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _plot_one(cm, model_name, output_dir, class_names):
    """
    Save one confusion-matrix PNG to `output_dir/cm_<model>.png`.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > thresh else "#333333",
            )

    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual", fontsize=11, labelpad=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"cm_{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_matrices(
    fitted_models,
    X_test,
    y_test,
    available_nb_vars,
    comparison,
    output_dir,
    class_names=("Negative (0)", "Positive (1)"),
):
    """
    Save one confusion-matrix PNG per fitted model into `output_dir`.
    """

    os.makedirs(output_dir, exist_ok=True)

    for model_name, model in fitted_models.items():
        if model_name == "Bernoulli Naive Bayes" and available_nb_vars:
            X_test_use = X_test[available_nb_vars]
        else:
            X_test_use = X_test

        threshold_match = comparison.loc[
            comparison["model"] == model_name, "threshold"
        ].values
        model_threshold = float(threshold_match[0]) if len(threshold_match) > 0 else 0.5

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_use)[:, 1]
            y_pred = (y_prob >= model_threshold).astype(int)
        else:
            y_pred = model.predict(X_test_use)

        cm = confusion_matrix(y_test, y_pred)
        _plot_one(cm, model_name, output_dir, class_names)

    print(f"All confusion matrices saved to {output_dir}/")
