import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append("model-comparison")
from functions import run_classification_models  # noqa: E402

from targets import TARGETS, KNOWLEDGE_VARS, NB_VARS, RAW_MISSING_MAP

DATA_PATH = "data/data.csv"
PLOTS_DIR = "visuals/plots"


def plot_confusion_matrix(cm, model_name, target_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

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
    ax.set_title(f"{target_name} — {model_name}", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"cm_{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_for_target(target, raw_df):
    name = target["name"]
    target_col = target["target_col"]
    output_dir = os.path.join(PLOTS_DIR, f"{name}_confusion_matrices")
    os.makedirs(output_dir, exist_ok=True)

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

    for model_name, model in fitted_models.items():
        X_test_use = X_test[NB_VARS] if model_name == "Bernoulli Naive Bayes" else X_test

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
        plot_confusion_matrix(cm, model_name, name, output_dir)

    print(f"Confusion matrices for {name} saved to {output_dir}/\n")


def main():
    raw_df = pd.read_csv(DATA_PATH)
    for target in TARGETS:
        print(f"=== {target['name']} ===")
        run_for_target(target, raw_df)


if __name__ == "__main__":
    main()
