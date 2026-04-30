import pandas as pd
import numpy as np

df = pd.read_csv("never-ready-final/data/data.csv")

# Keep valid responses
df = df[~df["SAVINGSRANGES"].isin([98, 99])].copy()

# Create binary target
# 0 = $0–$499 (codes 1–5)
# 1 = $500+ (codes 6+)
df["has_savings"] = (df["SAVINGSRANGES"] >= 5).astype(int)

print(df["has_savings"].value_counts(normalize=True))

from functions import run_classification_models

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

# Use thresholds learned from automatic tuning
tuned_thresholds = {
    "Baseline Logistic Regression": 0.38,
    "LASSO Logistic Regression with CV": 0.38,
    "Decision Tree": 0.33,
    "Random Forest": 0.27,
    "Bernoulli Naive Bayes": 0.27,
    "Gradient Boosting": 0.39
}

# Run models using fixed tuned thresholds
comparison, fitted_models = run_classification_models(
    df=df,
    target_col="has_savings",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="Has Savings ($500+)",
    test_size=0.25,
    random_state=42,
    thresholds=tuned_thresholds,
    tune_threshold=False,
    results_csv="never-ready-final/data/savings_results.csv"
)