"""Shared configuration for visualization scripts.

Defines the targets (fpl, fraud, low_HHI, high_HHI, savings, volatility)
that the bar chart, confusion matrix, and ROC AUC scripts iterate over.
"""

KNOWLEDGE_VARS = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct", "SUBKNOWL1",
]

# Bernoulli Naive Bayes uses the binary subset (no SUBKNOWL1).
NB_VARS = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct",
]

RAW_MISSING_MAP = {"SUBKNOWL1": [-1]}


def prep_fpl(df):
    df = df[df["fpl"].isin([1, 2, 3])].copy()
    df["low_fpl"] = df["fpl"].isin([1, 2]).astype(int)
    return df


def prep_fraud(df):
    df = df[df["FRAUD2"].isin([0, 1])].copy()
    df["fraud"] = df["FRAUD2"].astype(int)
    return df


def prep_hhi(df):
    df = df[df["PPINCIMP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()
    df["low_HHI"] = df["PPINCIMP"].isin([1, 2, 3]).astype(int)
    df["high_HHI"] = df["PPINCIMP"].isin([7, 8, 9]).astype(int)
    return df


def prep_savings(df):
    df = df[~df["SAVINGSRANGES"].isin([98, 99])].copy()
    df["has_savings"] = (df["SAVINGSRANGES"] >= 6).astype(int)
    return df


def prep_volatility(df):
    df = df[df["VOLATILITY"].isin([1, 2, 3])].copy()
    df["sbl_fin"] = df["VOLATILITY"].isin([1, 2]).astype(int)
    return df


TARGETS = [
    {
        "name": "fpl",
        "target_col": "low_fpl",
        "prep": prep_fpl,
        "results_csv": "data/fpl_results.csv",
        "threshold_metric": "balanced_accuracy",
    },
    {
        "name": "fraud",
        "target_col": "fraud",
        "prep": prep_fraud,
        "results_csv": "data/fraud_results.csv",
        "threshold_metric": "balanced_accuracy",
    },
    {
        "name": "low_HHI",
        "target_col": "low_HHI",
        "prep": prep_hhi,
        "results_csv": "data/low_HHI_results.csv",
        "threshold_metric": "balanced_accuracy",
    },
    {
        "name": "high_HHI",
        "target_col": "high_HHI",
        "prep": prep_hhi,
        "results_csv": "data/high_HHI_results.csv",
        "threshold_metric": "balanced_accuracy",
    },
    {
        "name": "savings",
        "target_col": "has_savings",
        "prep": prep_savings,
        "results_csv": "data/savings_results.csv",
        "threshold_metric": "f1",
    },
    {
        "name": "volatility",
        "target_col": "sbl_fin",
        "prep": prep_volatility,
        "results_csv": "data/volatility_results.csv",
        "threshold_metric": "balanced_accuracy",
    },
]
