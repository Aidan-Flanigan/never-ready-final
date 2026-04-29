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


def run_classification_models(
    df,
    target_col,
    predictor_vars,
    nb_vars=None,
    raw_missing_map=None,
    target_name=None,
    test_size=0.25,
    random_state=42
):
    """
    Runs logistic regression, LASSO CV, decision tree, random forest,
    and optional Bernoulli Naive Bayes for a binary classification target.

    Parameters:
    df : pandas DataFrame
        Dataset containing target and predictors.
    target_col : str
        Binary target column, coded 0/1.
    predictor_vars : list
        Main predictor variables used for logistic/tree/random forest models.
    nb_vars : list or None
        Binary variables for Bernoulli Naive Bayes.
        If None, Naive Bayes is skipped.
    raw_missing_map : dict or None
        Dictionary for special missing values.
        Example: {"SUBKNOWL1": [-1], "SAVINGSRANGES": [98, 99]}
    target_name : str or None
        Printed name for target. Defaults to target_col.
    test_size : float
        Test set share.
    random_state : int
        Random seed.

    Returns:
    comparison : pandas DataFrame
        Model comparison table.
    fitted_models : dict
        Dictionary of fitted model objects.
    extra_outputs : dict
        Coefficients and feature importance tables.
    """

    if target_name is None:
        target_name = target_col

    df_model = df.copy()

    # Keep only valid binary target observations
    df_model = df_model[df_model[target_col].isin([0, 1])].copy()

    # Keep only predictors that exist
    missing_predictors = [col for col in predictor_vars if col not in df_model.columns]
    available_predictors = [col for col in predictor_vars if col in df_model.columns]

    if missing_predictors:
        print("\nWarning: these predictors are missing and will be skipped:")
        print(missing_predictors)

    X = df_model[available_predictors].copy()
    y = df_model[target_col]

    # Replace special missing codes with np.nan
    if raw_missing_map is not None:
        for col, missing_values in raw_missing_map.items():
            if col in X.columns:
                X[col] = X[col].replace(missing_values, np.nan)

    print("\n================ Setup ================")
    print("Target:", target_name)

    print("\nPredictors used:")
    print(available_predictors)

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    results = []
    fitted_models = {}
    extra_outputs = {}

    def evaluate_model(model, model_name, X_train_use, X_test_use):
        model.fit(X_train_use, y_train)

        y_pred = model.predict(X_test_use)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_use)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        print(f"\n================ {model_name} ================")

        print("\nModel performance:")
        print("Accuracy:", round(accuracy, 3))
        print("Balanced accuracy:", round(balanced_acc, 3))
        print("ROC AUC:", round(roc_auc, 3))
        print(f"Precision for {target_name}=1:", round(precision_pos, 3))
        print(f"Recall for {target_name}=1:", round(recall_pos, 3))
        print(f"F1 for {target_name}=1:", round(f1_pos, 3))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "balanced_accuracy": balanced_acc,
            f"precision_{target_name}": precision_pos,
            f"recall_{target_name}": recall_pos,
            f"f1_{target_name}": f1_pos
        })

        fitted_models[model_name] = model

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
        "variable": available_predictors,
        "coefficient": logit.coef_[0]
    }).sort_values("coefficient", ascending=False)

    print("\nBaseline logistic regression coefficients:")
    print(coef_table)

    extra_outputs["baseline_coefficients"] = coef_table

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
            random_state=random_state
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
        "variable": available_predictors,
        "coefficient": lasso_logit.coef_[0]
    }).sort_values("coefficient", ascending=False)

    selected_vars = lasso_coef_table[lasso_coef_table["coefficient"] != 0]

    print("\nLASSO CV coefficients:")
    print(lasso_coef_table)

    print("\nLASSO CV selected variables:")
    print(selected_vars)

    print("\nNumber of variables selected by LASSO CV:")
    print(len(selected_vars), "out of", len(available_predictors))

    print("\nBest C chosen by cross-validation:")
    print(lasso_logit.C_[0])

    extra_outputs["lasso_coefficients"] = lasso_coef_table
    extra_outputs["lasso_selected_variables"] = selected_vars
    extra_outputs["lasso_best_c"] = lasso_logit.C_[0]

    # Model 3: Decision Tree
    decision_tree = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state
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
        "variable": available_predictors,
        "importance": tree.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nDecision tree feature importance:")
    print(tree_importance)

    extra_outputs["tree_importance"] = tree_importance

    # Model 4: Random Forest
    random_forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=random_state
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
        "variable": available_predictors,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nRandom forest feature importance:")
    print(rf_importance)

    extra_outputs["random_forest_importance"] = rf_importance

    # Model 5: Bernoulli Naive Bayes
    if nb_vars is not None:
        missing_nb_vars = [col for col in nb_vars if col not in df_model.columns]
        available_nb_vars = [col for col in nb_vars if col in df_model.columns]

        if missing_nb_vars:
            print("\nWarning: these Naive Bayes variables are missing and will be skipped:")
            print(missing_nb_vars)

        if len(available_nb_vars) > 0:
            X_nb = df_model[available_nb_vars].copy()

            X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
                X_nb, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            assert y_train_nb.equals(y_train)
            assert y_test_nb.equals(y_test)

            naive_bayes = BernoulliNB()

            naive_bayes = evaluate_model(
                naive_bayes,
                "Bernoulli Naive Bayes",
                X_train_nb,
                X_test_nb
            )
        else:
            print("\nSkipping Naive Bayes because no valid nb_vars were found.")

    # Final comparison
    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    print("\n================ Final Model Comparison ================")
    print(comparison.round(3))

    return comparison, fitted_models, extra_outputs