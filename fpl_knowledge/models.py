import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv("data/data.csv")

# target: fpl
# 1 = <100% FPL
# 2 = 100%-199% FPL
# 3 = 200%+ FPL

# Binary target:
# 1 = below 200% FPL
# 0 = 200%+ FPL
df = df[df["fpl"].isin([1, 2, 3])].copy()
df["low_fpl"] = df["fpl"].isin([1, 2]).astype(int)


# Predictors
knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct", "SUBKNOWL1"
]

# Naive Bayes uses only binary vars
nb_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct"
]

X = df[knowledge_vars].copy()

# FK/KH correct variables already code refused answers as 0 = incorrect.
# SUBKNOWL1 is raw self-rated knowledge, where -1 means refused.
X["SUBKNOWL1"] = X["SUBKNOWL1"].replace(-1, np.nan)

y = df["low_fpl"]

print("Predictors used:")
print(knowledge_vars)

print("\nTarget distribution:")
print(y.value_counts(normalize=True))


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

results = []

def evaluate_model(model, model_name, X_train_use, X_test_use):
    model.fit(X_train_use, y_train)

    y_pred = model.predict(X_test_use)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_use)[:, 1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision_low_fpl = precision_score(y_test, y_pred, pos_label=1)
    recall_low_fpl = recall_score(y_test, y_pred, pos_label=1)
    f1_low_fpl = f1_score(y_test, y_pred, pos_label=1)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    print(f"\n================ {model_name} ================")

    print("\nModel performance:")
    print("Accuracy:", round(accuracy, 3))
    print("Balanced accuracy:", round(balanced_acc, 3))
    print("ROC AUC:", round(roc_auc, 3))
    print("Precision for low_fpl:", round(precision_low_fpl, 3))
    print("Recall for low_fpl:", round(recall_low_fpl, 3))
    print("F1 for low_fpl:", round(f1_low_fpl, 3))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "precision_low_fpl": precision_low_fpl,
        "recall_low_fpl": recall_low_fpl,
        "f1_low_fpl": f1_low_fpl
    })

    return model


# Model 1: Baseline Logistic Regression
baseline_logit = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

baseline_logit = evaluate_model(
    baseline_logit,
    "Baseline Logistic Regression",
    X_train,
    X_test
)

logit = baseline_logit.named_steps["logisticregression"]

coef_table = pd.DataFrame({
    "variable": knowledge_vars,
    "coefficient": logit.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nBaseline logistic regression coefficients:")
print(coef_table)


# Model 2: LASSO Logistic Regression with CV
lasso_cv = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        Cs=20,
        cv=5,
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

lasso_cv = evaluate_model(
    lasso_cv,
    "LASSO Logistic Regression with CV",
    X_train,
    X_test
)

lasso_logit = lasso_cv.named_steps["logisticregressioncv"]

lasso_coef_table = pd.DataFrame({
    "variable": knowledge_vars,
    "coefficient": lasso_logit.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nLASSO CV coefficients:")
print(lasso_coef_table)

selected_vars = lasso_coef_table[lasso_coef_table["coefficient"] != 0]

print("\nLASSO CV selected variables:")
print(selected_vars)

print("\nNumber of variables selected by LASSO CV:")
print(len(selected_vars), "out of", len(knowledge_vars))

print("\nBest C chosen by cross-validation:")
print(lasso_logit.C_[0])


# Model 3: Decision Tree
decision_tree = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    )
)

decision_tree = evaluate_model(
    decision_tree,
    "Decision Tree",
    X_train,
    X_test
)

tree = decision_tree.named_steps["decisiontreeclassifier"]

tree_importance = pd.DataFrame({
    "variable": knowledge_vars,
    "importance": tree.feature_importances_
}).sort_values("importance", ascending=False)

print("\nDecision tree feature importance:")
print(tree_importance)



# Model 4: Random Forest
random_forest = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    )
)

random_forest = evaluate_model(
    random_forest,
    "Random Forest",
    X_train,
    X_test
)

rf = random_forest.named_steps["randomforestclassifier"]

rf_importance = pd.DataFrame({
    "variable": knowledge_vars,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nRandom forest feature importance:")
print(rf_importance)



# Model 5: Bernoulli Naive Bayes
# BernoulliNB is designed for binary variables.
# So we use only the individual correct/incorrect variables,
# not LMscore, KHscore, or SUBKNOWL1.
X_nb = df[nb_vars].copy()

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Sanity check: same y split as before
assert y_train_nb.equals(y_train)
assert y_test_nb.equals(y_test)

naive_bayes = BernoulliNB()

naive_bayes = evaluate_model(
    naive_bayes,
    "Bernoulli Naive Bayes",
    X_train_nb,
    X_test_nb
)

# Final comparison
comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

print("\n================ Final Model Comparison ================")
print(comparison.round(3))