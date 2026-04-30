
import pandas as pd
from functions import run_classification_models

# Load data
df = pd.read_csv("data/data.csv")

# Target: Volatility
# Question: Which of the following best describes how your household’s income 
#           changes from month to month, if at all?
# 1 = Roughly the same each month
# 2 = Roughly the same most months, but some unusually high or low months 
#     during the year
# 3 = Often varies quite a bit from one month to the next

# Binary target: Stable Finances
# 1 = Roughly the same for most or each months (Codes 1, 2)
# 0 = Often varies quite a bit from one month to the next (Code 3)
df = df[df["VOLATILITY"].isin([1, 2, 3])]
df["sbl_fin"] = df["VOLATILITY"].isin([1, 2]).astype(int)

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
comparison, fitted_models = run_classification_models(
    df=df,
    target_col="sbl_fin",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="sbl_fin",
    test_size=0.25,
    random_state=42,
    tune_threshold=True,
    threshold_metric="balanced_accuracy",
    results_csv="data/volatility_results.csv"
)

