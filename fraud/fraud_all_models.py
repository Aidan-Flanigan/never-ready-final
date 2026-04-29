import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Data Prep 
df = pd.read_csv("data/data.csv")
df = df[df["FRAUD2"].isin([0, 1])].copy()
df["fraud"] = df["FRAUD2"].astype(int)

knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct", "KH1correct", "KH2correct", "KH3correct",
    "KH4correct", "KH5correct", "KH6correct", "KH7correct", "KH8correct", "KH9correct",
    "ON1correct", "ON2correct", "SUBKNOWL1"
]
nb_vars = knowledge_vars[:12]

X = df[knowledge_vars].copy()
X["SUBKNOWL1"] = X["SUBKNOWL1"].replace(-1, np.nan)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Models
models = {
    "Baseline Logit": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "LASSO CV": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegressionCV(penalty="l1", solver="liblinear", cv=5, class_weight="balanced", random_state=42)),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, max_depth=5, class_weight="balanced", random_state=42)),
    "Gradient Boost": make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    "Naive Bayes": BernoulliNB()
}

# Loops
results = []
for name, model in models.items():
    xtr, xte = (X_train[nb_vars], X_test[nb_vars]) if name == "Naive Bayes" else (X_train, X_test)
    
    model.fit(xtr, y_train)
    preds = model.predict(xte)
    probs = model.predict_proba(xte)[:, 1] if hasattr(model, "predict_proba") else None
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, probs) if probs is not None else np.nan
    })

# Summary
print(pd.DataFrame(results).sort_values("ROC AUC", ascending=False).round(3))