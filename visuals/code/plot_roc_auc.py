import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

sys.path.append("model-comparison")
from functions import run_classification_models

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/data.csv"
OUTPUT_DIR  = "visuals/plots/fraud_roc_auc_curves"
OUTPUT_FILE = "roc_auc_curves.png"
TARGET_COL  = "fraud"
TARGET_NAME = "fraud"
# ─────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data prep
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

# Run models
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

# Rebuild test set
df_model = df[df[TARGET_COL].isin([0, 1])].copy()
X = df_model[knowledge_vars].copy()
X["SUBKNOWL1"] = X["SUBKNOWL1"].replace([-1], np.nan)
y = df_model[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Colors per model ──────────────────────────────────────────────────────
colors = [
    "#2E86AB", "#E84855", "#3BB273",
    "#F4A261", "#9B5DE5", "#00BBF9"
]

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

for (model_name, model), color in zip(fitted_models.items(), colors):
    if not hasattr(model, "predict_proba"):
        continue

    X_test_use = X_test[nb_vars] if model_name == "Bernoulli Naive Bayes" else X_test

    y_prob = model.predict_proba(X_test_use)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Get the selected threshold for this model
    model_threshold = comparison.loc[
        comparison["model"] == model_name, "threshold"
    ].values
    model_threshold = float(model_threshold[0]) if len(model_threshold) > 0 else 0.5

    # Find the point on the curve closest to the selected threshold
    diff = np.abs(thresholds - model_threshold)
    threshold_idx = np.argmin(diff)

    ax.plot(
        fpr, tpr,
        color=color,
        linewidth=2,
        label=f"{model_name}  (AUC = {roc_auc:.3f})"
    )

    # Mark the selected threshold on the curve
    ax.scatter(
        fpr[threshold_idx], tpr[threshold_idx],
        color=color, s=80, zorder=5, edgecolors="white", linewidths=1.2
    )

# Diagonal reference line
ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", linewidth=1.2, label="Random classifier")

ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
ax.set_title("ROC Curves by Model", fontsize=14, fontweight="bold", pad=15)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# Legend — models + note about dots
dot_note = mlines.Line2D(
    [], [], color="gray", marker="o", linestyle="None",
    markersize=7, markeredgecolor="white",
    label="● = selected threshold"
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles + [dot_note],
    fontsize=9, framealpha=0.6,
    loc="lower right"
)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"ROC curve saved to {save_path}")
