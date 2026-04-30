import pandas as pd
from functions import run_classification_models

# Load data
df = pd.read_csv("data/data.csv")



df = df[df["PPINCIMP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()
df["low_HHI"] = df["PPINCIMP"].isin([1, 2, 3]).astype(int)
df["high_HHI"] = df["PPINCIMP"].isin([7, 8, 9]).astype(int)


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
    "Baseline Logistic Regression": 0.5,
    "LASSO Logistic Regression with CV": 0.5,
    "Decision Tree": 0.5,
    "Random Forest": 0.5,
    "Bernoulli Naive Bayes": 0.5,
    "Gradient Boosting": 0.5
}

# Run models
print("testing models with low_fpl")
comparison, fitted_models = run_classification_models(
    df=df,
    target_col="low_HHI",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="low_HHI",
    test_size=0.25,
    random_state=42,
    tune_threshold=True,
    threshold_metric="balanced_accuracy",
    results_csv="data/low_HHI_results.csv"
)   

print("testing models with high_fpl")
comparison, fitted_models = run_classification_models(
    df=df,
    target_col="high_HHI",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="high_HHI",
    test_size=0.25,
    random_state=42,
    tune_threshold=True,
    threshold_metric="balanced_accuracy",
    results_csv="data/high_HHI_results.csv"
)