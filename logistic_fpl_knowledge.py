import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("data.csv")

# target: fpl
# 1 = <100% FPL
# 2 = 100%-199% FPL
# 3 = 200%+ FPL

# Define Binary Target:
# 1 = below 200% FPL
# 0 = 200%+ FPL
df = df[df["fpl"].isin([1, 2, 3])].copy()
df["low_fpl"] = df["fpl"].isin([1, 2]).astype(int)

knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct",
    "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct",
    "KH7correct", "KH8correct", "KH9correct",
    "LMscore", "KHscore", "SUBKNOWL1"
]

X = df[knowledge_vars]
y = df["low_fpl"]

print("Predictors used:")
print(knowledge_vars)
print("\nTarget distribution:")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y

)

# Simple Logistics Regression
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced")

)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\nModel performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 3))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Coefficients
logit = model.named_steps["logisticregression"]
coef_table = pd.DataFrame({
    "variable": knowledge_vars,
    "coefficient": logit.coef_[0]
}).sort_values("coefficient", ascending=False)
print("\nLogistic regression coefficients:")
print(coef_table)