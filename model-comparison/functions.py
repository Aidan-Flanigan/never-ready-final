import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from catboost import CatBoostClassifier
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
    random_state=42,
    threshold=0.5,
    thresholds=None,
    tune_threshold=False,
    threshold_metric="balanced_accuracy",
    cv_threshold_folds=5,
    results_csv="data/model_results.csv"
):
    """
    Train all classifiers on a binary target and return fitted models, the
    held-out test set, and a comparison table. Threshold can be auto-tuned via
    CV on the training set or supplied per model via `thresholds`.
    """

    if target_name is None:
        target_name = target_col

    df_model = df.copy()
    df_model = df_model[df_model[target_col].isin([0, 1])].copy()

    missing_predictors = [col for col in predictor_vars if col not in df_model.columns]
    available_predictors = [col for col in predictor_vars if col in df_model.columns]

    if missing_predictors:
        print("\nWarning: these predictors are missing and will be skipped:")
        print(missing_predictors)

    X = df_model[available_predictors].copy()
    y = df_model[target_col]

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
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    results = []
    fitted_models = {}

    def find_best_threshold(y_true, y_prob, metric="balanced_accuracy"):
        """
        Sweep thresholds in [0.05, 0.95] and return the one that maximizes `metric`.
        """

        candidate_thresholds = np.arange(0.05, 0.96, 0.01)

        best_threshold = 0.5
        best_score = -1

        for t in candidate_thresholds:
            y_pred_t = (y_prob >= t).astype(int)

            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, y_pred_t)
            elif metric == "f1":
                score = f1_score(y_true, y_pred_t, pos_label=1, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred_t, pos_label=1, zero_division=0)
            else:
                raise ValueError(
                    "metric must be 'balanced_accuracy', 'f1', or 'recall'"
                )

            if score > best_score:
                best_score = score
                best_threshold = t

        return best_threshold, best_score

    def evaluate_model(model, model_name, X_train_use, X_test_use):
        """
        Fit, optionally CV-tune the threshold on the training set, then score once on the test set.
        """

        model_threshold = threshold
        threshold_score = None

        if thresholds is not None:
            model_threshold = thresholds.get(model_name, threshold)

        if tune_threshold and hasattr(model, "predict_proba"):
            cv = StratifiedKFold(
                n_splits=cv_threshold_folds,
                shuffle=True,
                random_state=random_state
            )

            y_train_prob_cv = cross_val_predict(
                model,
                X_train_use,
                y_train,
                cv=cv,
                method="predict_proba"
            )[:, 1]

            model_threshold, threshold_score = find_best_threshold(
                y_train,
                y_train_prob_cv,
                metric=threshold_metric
            )

        model.fit(X_train_use, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_use)[:, 1]
            y_pred = (y_prob >= model_threshold).astype(int)
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            y_prob = None
            y_pred = model.predict(X_test_use)
            roc_auc = np.nan
            model_threshold = None

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        print(f"\n================ {model_name} ================")
        print("Threshold used:", model_threshold)

        if threshold_score is not None:
            print(f"CV threshold metric ({threshold_metric}):", round(threshold_score, 3))

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
            "threshold": model_threshold,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "balanced_accuracy": balanced_acc,
            "precision": precision_pos,
            "recall": recall_pos,
            "f1": f1_pos
        })

        pd.DataFrame(results).round(4).to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

        fitted_models[model_name] = model

        return model

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

    available_nb_vars = []
    if nb_vars is not None:
        missing_nb_vars = [col for col in nb_vars if col not in df_model.columns]
        available_nb_vars = [col for col in nb_vars if col in df_model.columns]

        if missing_nb_vars:
            print("\nWarning: these Naive Bayes variables are missing and will be skipped:")
            print(missing_nb_vars)

        if len(available_nb_vars) > 0:
            X_nb = df_model[available_nb_vars].copy()

            X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
                X_nb,
                y,
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

    gradient_boosting = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=30,
            random_state=random_state
        )
    )

    gradient_boosting = evaluate_model(
        gradient_boosting,
        "Gradient Boosting",
        X_train,
        X_test
    )

    gb = gradient_boosting.named_steps["gradientboostingclassifier"]

    gb_importance = pd.DataFrame({
        "variable": available_predictors,
        "importance": gb.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nGradient boosting feature importance:")
    print(gb_importance)
    
    catboost_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False
    )

    catboost_model = evaluate_model(
        catboost_model,
        "CatBoost",
        X_train,
        X_test
    )

    importance = pd.DataFrame({
        "variable": available_predictors,
        "importance": catboost_model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    print("\nCatBoost feature importance:")
    print(importance)
 
    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    print("\n================ Final Model Comparison ================")
    print(comparison.round(3))

    return {
        "comparison": comparison,
        "fitted_models": fitted_models,
        "X_test": X_test,
        "y_test": y_test,
        "available_predictors": available_predictors,
        "available_nb_vars": available_nb_vars,
    }