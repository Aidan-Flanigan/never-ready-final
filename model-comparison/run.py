"""
Run all problems:
    python model-comparison/run.py

To target a single classification problem:
    python model-comparison/run.py fpl fraud

Available target names:
    fpl, fraud, low_HHI, high_HHI, savings, volatility
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(HERE)
sys.path.append(os.path.join(PROJECT_ROOT, "visuals", "code"))

from functions import run_classification_models  
from targets import (  
    TARGETS,
    KNOWLEDGE_VARS,
    NB_VARS,
    RAW_MISSING_MAP,
)
from bar_chart import plot_metric_bar_chart 
from plot_confusion_matrices import plot_confusion_matrices 
from plot_roc_auc import plot_roc_curves 

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "visuals", "plots")


def run_target(target, raw_df):
    """Train every model on one target and emit its CSV, bar chart, CMs, and ROC plot."""
    name = target["name"]
    print(f"\n############### {name} ###############")

    df = target["prep"](raw_df)

    result = run_classification_models(
        df=df,
        target_col=target["target_col"],
        predictor_vars=KNOWLEDGE_VARS,
        nb_vars=NB_VARS,
        raw_missing_map=RAW_MISSING_MAP,
        target_name=name,
        test_size=0.25,
        random_state=42,
        tune_threshold=target.get("tune_threshold", True),
        threshold_metric=target.get("threshold_metric", "balanced_accuracy"),
        thresholds=target.get("thresholds"),
        results_csv=os.path.join(PROJECT_ROOT, target["results_csv"]),
    )

    bar_path = os.path.join(PLOTS_DIR, f"{name}_comparison.png")
    plot_metric_bar_chart(
        results_csv=os.path.join(PROJECT_ROOT, target["results_csv"]),
        output_path=bar_path,
        target_name=name,
    )

    cm_dir = os.path.join(PLOTS_DIR, f"{name}_confusion_matrices")
    plot_confusion_matrices(
        fitted_models=result["fitted_models"],
        X_test=result["X_test"],
        y_test=result["y_test"],
        available_nb_vars=result["available_nb_vars"],
        comparison=result["comparison"],
        output_dir=cm_dir,
        class_names=target.get("class_names", ("Negative (0)", "Positive (1)")),
    )

    roc_path = os.path.join(PLOTS_DIR, f"{name}_roc_auc.png")
    plot_roc_curves(
        fitted_models=result["fitted_models"],
        X_test=result["X_test"],
        y_test=result["y_test"],
        available_nb_vars=result["available_nb_vars"],
        comparison=result["comparison"],
        target_name=name,
        output_path=roc_path,
    )


def main(argv):
    """Run every target in TARGETS, or only the names passed on the CLI."""
    selected = set(argv[1:])
    raw_df = pd.read_csv(DATA_PATH)

    targets_to_run = [t for t in TARGETS if not selected or t["name"] in selected]

    if selected and not targets_to_run:
        valid = ", ".join(t["name"] for t in TARGETS)
        raise SystemExit(f"No matching targets. Valid names: {valid}")

    for target in targets_to_run:
        run_target(target, raw_df)


if __name__ == "__main__":
    main(sys.argv)
