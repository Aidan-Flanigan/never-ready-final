import pandas as pd
from functions import run_classification_models

# Load data
df = pd.read_csv("data/data.csv")

# Define current target: low_fpl
# fpl coding:
# 1 = <100% FPL
# 2 = 100%-199% FPL
# 3 = 200%+ FPL
#
# low_fpl:
# 1 = below 200% FPL
# 0 = 200%+ FPL

df = df[df["FRAUD2"].isin([0, 1])].copy()
df["fraud"] = df["FRAUD2"].astype(int)


# Define predictors
knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct", "SUBKNOWL1"
]


# Naive Bayes predictors
# Bernoulli Naive Bayes should use binary variables only.
nb_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct", 
    "ON1correct", "ON2correct"
]

thresholds = {
    "Baseline Logistic Regression": 0.25,
    "LASSO Logistic Regression with CV": 0.25,
    "Decision Tree": 0.30,
    "Random Forest": 0.25,
    "Bernoulli Naive Bayes": 0.25,
    "Gradient Boosting": 0.15
}

# Run models
comparison, fitted_models = run_classification_models(
    df=df,
    target_col="fraud",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="fraud",
    test_size=0.25,
    random_state=42,
    tune_threshold=True,
    threshold_metric="balanced_accuracy"
)
