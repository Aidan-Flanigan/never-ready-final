"""
Per-target config consumed by run_all.py. Edit a TARGETS entry to change
predictors, prep, threshold tuning, or class labels for that target.
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
    """
    low_fpl = 1 if FPL is below 200%, else 0.
    """
    df = df[df["fpl"].isin([1, 2, 3])].copy()
    df["low_fpl"] = df["fpl"].isin([1, 2]).astype(int)
    return df


def prep_fraud(df):
    """
    fraud = FRAUD2 cast to 0/1.
    """
    df = df[df["FRAUD2"].isin([0, 1])].copy()
    df["fraud"] = df["FRAUD2"].astype(int)
    return df


def prep_hhi(df):
    """
    low_HHI = bottom 3 PPINCIMP brackets; high_HHI = top 3
    """
    df = df[df["PPINCIMP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()
    df["low_HHI"] = df["PPINCIMP"].isin([1, 2, 3]).astype(int)
    df["high_HHI"] = df["PPINCIMP"].isin([7, 8, 9]).astype(int)
    return df


def prep_savings(df):
    """
    has_savings = 1 if SAVINGSRANGES >= 6 ($500+), else 0.
    """
    df = df[~df["SAVINGSRANGES"].isin([98, 99])].copy()
    df["has_savings"] = (df["SAVINGSRANGES"] >= 6).astype(int)
    return df


def prep_volatility(df):
    """
    sbl_fin = 1 if income is roughly stable month-to-month, else 0.
    """
    df = df[df["VOLATILITY"].isin([1, 2, 3])].copy()
    df["sbl_fin"] = df["VOLATILITY"].isin([1, 2]).astype(int)
    return df


TARGETS = [
    {
        "name": "fpl",
        "target_col": "low_fpl",
        "prep": prep_fpl,
        "results_csv": "data/fpl_results.csv",
        "tune_threshold": True,
        "threshold_metric": "balanced_accuracy",
        "thresholds": None,
        "class_names": ("Above 200% FPL (0)", "Below 200% FPL (1)"),
    },
    {
        "name": "fraud",
        "target_col": "fraud",
        "prep": prep_fraud,
        "results_csv": "data/fraud_results.csv",
        "tune_threshold": True,
        "threshold_metric": "balanced_accuracy",
        "thresholds": None,
        "class_names": ("Not Fraud (0)", "Fraud (1)"),
    },
    {
        "name": "low_HHI",
        "target_col": "low_HHI",
        "prep": prep_hhi,
        "results_csv": "data/low_HHI_results.csv",
        "tune_threshold": True,
        "threshold_metric": "balanced_accuracy",
        "thresholds": None,
        "class_names": ("Not Low Income (0)", "Low Income (1)"),
    },
    {
        "name": "high_HHI",
        "target_col": "high_HHI",
        "prep": prep_hhi,
        "results_csv": "data/high_HHI_results.csv",
        "tune_threshold": True,
        "threshold_metric": "balanced_accuracy",
        "thresholds": None,
        "class_names": ("Not High Income (0)", "High Income (1)"),
    },
    {
        "name": "savings",
        "target_col": "has_savings",
        "prep": prep_savings,
        "results_csv": "data/savings_results.csv",
        "tune_threshold": True,
        "threshold_metric": "f1",
        "thresholds": None,
        "class_names": ("Below $500 (0)", "$500 or more (1)"),
    },
    {
        "name": "volatility",
        "target_col": "sbl_fin",
        "prep": prep_volatility,
        "results_csv": "data/volatility_results.csv",
        "tune_threshold": True,
        "threshold_metric": "balanced_accuracy",
        "thresholds": None,
        "class_names": ("Unstable Income (0)", "Stable Income (1)"),
    },
]
