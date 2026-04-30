import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import sys
sys.path.append("model-comparison")
from functions import run_classification_models

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/data.csv"
OUTPUT_DIR  = "visuals/plots/fraud_confusion_matrices.png"
TARGET_COL  = "fraud"
TARGET_NAME = "fraud"
# ─────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prep data
df = pd.read_csv(DATA_PATH)
df = df[df["FRAUD2"].isin([0, 1])].copy()
df["fraud"] = df["FRAUD2"].astype(int)

knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct", "SUBKNOWL1"
]

nb_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct"
]

thresholds = {
    "Baseline Logistic Regression":      0.25,
    "LASSO Logistic Regression with CV": 0.25,
    "Decision Tree":                     0.30,
    "Random Forest":                     0.25,
    "Bernoulli Naive Bayes":             0.25,
    "Gradient Boosting":                 0.15
}

# Run models to get fitted_models back
comparison, fitted_models = run_classification_models(
    df=df,
    target_col=TARGET_COL,
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name=TARGET_NAME,
    test_size=0.25,
    random_state=42,
    tune_threshold=True,
    threshold_metric="balanced_accuracy",
    results_csv="data/model_results.csv"
)

# Rebuild test set the same way so predictions align
df_model = df[df[TARGET_COL].isin([0, 1])].copy()
X = df_model[knowledge_vars].copy()
X["SUBKNOWL1"] = X["SUBKNOWL1"].replace([-1], np.nan)
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Plot one confusion matrix per model ───────────────────────────────────
def plot_confusion_matrix(cm, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Not Fraud (0)", "Fraud (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > thresh else "#333333"
            )

    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual", fontsize=11, labelpad=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"cm_{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


for model_name, model in fitted_models.items():
    # Use nb feature set for Naive Bayes, full set for everything else
    if model_name == "Bernoulli Naive Bayes":
        X_test_use = X_test[nb_vars]
    else:
        X_test_use = X_test

    threshold = comparison.loc[comparison["model"] == model_name, "threshold"].values
    model_threshold = float(threshold[0]) if len(threshold) > 0 else 0.5

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_use)[:, 1]
        y_pred = (y_prob >= model_threshold).astype(int)
    else:
        y_pred = model.predict(X_test_use)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, OUTPUT_DIR)

print(f"\nAll confusion matrices saved to {OUTPUT_DIR}/")
