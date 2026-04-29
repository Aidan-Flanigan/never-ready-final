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

df = df[df["fpl"].isin([1, 2, 3])].copy()
df["low_fpl"] = df["fpl"].isin([1, 2]).astype(int)


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
# So do not include LMscore, KHscore, or SUBKNOWL1.

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
    target_col="low_fpl",
    predictor_vars=knowledge_vars,
    nb_vars=nb_vars,
    raw_missing_map={"SUBKNOWL1": [-1]},
    target_name="low_fpl",
    test_size=0.25,
    random_state=42
)

