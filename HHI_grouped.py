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
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score
)

# --------------------------------------------------
# 1. Load and Prep Data
# --------------------------------------------------
DATA_PATH = 'data/data.csv'
df = pd.read_csv(DATA_PATH)

# Target Engineering
df = df[df["PPINCIMP"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()
df["low_HHI"] = df["PPINCIMP"].isin([1, 2, 3]).astype(int)
df["high_HHI"] = df["PPINCIMP"].isin([7, 8, 9]).astype(int)

knowledge_vars = [
    "FK1correct", "FK2correct", "FK3correct", "KH1correct", "KH2correct", 
    "KH3correct", "KH4correct", "KH5correct", "KH6correct", "KH7correct", 
    "KH8correct", "KH9correct", "LMscore", "KHscore", "SUBKNOWL1"
]

nb_vars = [
    "FK1correct", "FK2correct", "FK3correct", "KH1correct", "KH2correct", 
    "KH3correct", "KH4correct", "KH5correct", "KH6correct", "KH7correct", 
    "KH8correct", "KH9correct"
]

X = df[knowledge_vars].copy()
X["SUBKNOWL1"] = X["SUBKNOWL1"].replace(-1, np.nan)

# --------------------------------------------------
# 2. Evaluation Function (Matching your template)
# --------------------------------------------------
def evaluate_model(model, model_name, target_name, X_train_use, X_test_use, y_train, y_test, results_list):
    model.fit(X_train_use, y_train)
    y_pred = model.predict(X_test_use)
    y_prob = model.predict_proba(X_test_use)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    metrics = {
        "target": target_name,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    }

    print(f"\n================ {model_name} ({target_name}) ================")
    print(f"Accuracy: {metrics['accuracy']:.3f} | ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    results_list.append(metrics)
    return model

# --------------------------------------------------
# 3. Main Modeling Loop
# --------------------------------------------------
all_results = []

for target_col in ["low_HHI", "high_HHI"]:
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\n\n" + "#"*60)
    print(f"### ANALYSIS FOR TARGET: {target_col} ###")
    print("#"*60)

    # Model 1: Logistic Regression
    lr_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), 
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
    evaluate_model(lr_pipe, "Logistic Regression", target_col, X_train, X_test, y_train, y_test, all_results)
    
    # Model 2: LASSO CV
    lasso_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), 
                               LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=20, cv=5, 
                                                    scoring="roc_auc", class_weight="balanced", random_state=42))
    evaluate_model(lasso_pipe, "LASSO CV", target_col, X_train, X_test, y_train, y_test, all_results)
    
    # Coefficients for LASSO
    lasso_model = lasso_pipe.named_steps['logisticregressioncv']
    lasso_coefs = pd.DataFrame({"variable": knowledge_vars, "coef": lasso_model.coef_[0]}).sort_values("coef", ascending=False)
    print(f"\nLASSO Selected Vars ({target_col}):\n", lasso_coefs[lasso_coefs['coef'] != 0])

    # Model 3: Decision Tree
    dt_pipe = make_pipeline(SimpleImputer(strategy="median"), 
                            DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced", random_state=42))
    evaluate_model(dt_pipe, "Decision Tree", target_col, X_train, X_test, y_train, y_test, all_results)

    # Model 4: Random Forest
    rf_pipe = make_pipeline(SimpleImputer(strategy="median"), 
                            RandomForestClassifier(n_estimators=500, max_depth=5, class_weight="balanced", random_state=42))
    evaluate_model(rf_pipe, "Random Forest", target_col, X_train, X_test, y_train, y_test, all_results)

    # Model 5: Bernoulli Naive Bayes (subsetting features)
    nb_model = make_pipeline(SimpleImputer(strategy="most_frequent"), BernoulliNB())
    evaluate_model(nb_model, "Bernoulli NB", target_col, X_train[nb_vars], X_test[nb_vars], y_train, y_test, all_results)

# --------------------------------------------------
# 4. Final Comparison Table
# --------------------------------------------------
comparison_df = pd.DataFrame(all_results)
print("\n" + "="*25 + " FINAL COMPARISON " + "="*25)
# Sorting by Target then ROC AUC
print(comparison_df.sort_values(["target", "roc_auc"], ascending=[True, False]).to_string(index=False))