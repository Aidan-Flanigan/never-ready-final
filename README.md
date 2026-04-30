# Using Financial Knowledge Measures to Predict Household Financial Vulnerability
# I. Problem Statement and Motivation
This project attempts to explore whether financial knowledge measures can predict different forms of household financial vulnerability, including poverty status, fraud exposure, household income, savings, and income volatility. The motivation is that financial well-being is not only about income, but also about whether households can manage day-to-day finances, absorb shocks, and avoid harmful financial outcomes. The CFPB National Financial Well-Being Survey is well suited for this question because it includes measures of financial knowledge, financial skill, financial behavior, income, savings, safety nets, and financial experiences for a national sample of U.S. adults. In this project, we use financial knowledge, numeracy, and self-assessed knowledge variables to build classification models for several vulnerability outcomes, then compare whether interpretable models such as logistic regression perform similarly to more flexible machine learning models such as random forests and gradient boosting

## II. Data

### Data Source and Collection

For this project, we use the Consumer Financial Protection Bureau’s National Financial Well-Being Survey Public Use File. The survey was conducted online in English and Spanish between October and December 2016 and includes 6,394 completed responses from U.S. adults. The sample was drawn from GfK’s KnowledgePanel and was designed to represent the adult population of the 50 U.S. states and Washington, D.C., with an additional oversample of adults aged 62 and older. The dataset includes survey responses collected by the CFPB as well as pre-existing panel information on respondents, such as household income, poverty status, and demographic characteristics.

### Data Summary and Relevant Variables

Our analysis focuses on whether financial knowledge measures can predict several household financial vulnerability outcomes. The main predictor variables include the Lusardi-Mitchell financial knowledge items (`FK1correct`, `FK2correct`, `FK3correct`), the Knoll-Houts financial knowledge items (`KH1correct` to `KH9correct`), objective numeracy items (`ON1correct`, `ON2correct`), and self-assessed financial knowledge (`SUBKNOWL1`). The target variables are constructed as binary classification outcomes, including poverty status, fraud exposure, household income, savings level, and income volatility. These outcomes capture different dimensions of financial vulnerability, from low income and low savings to unstable income and exposure to financial fraud.

A more detailed description of the dataset, survey design, variable definitions, and codingA more detailed description of the dataset, survey design, variable definitions, and coding rules can be found in the official CFPB documentation and codebook, saved in this repository as `data/user_guide.pdf`.

### Limitations of Data

The data are cross-sectional, so the models should be interpreted as predictive rather than causal. The survey measures associations between financial knowledge and financial vulnerability, but it cannot prove that financial knowledge causes better or worse financial outcomes. Some variables are self-reported, which may introduce measurement error or recall bias, especially for sensitive topics such as income, savings, and fraud exposure. The dataset also includes nonsubstantive response codes such as refusal, “don’t know,” and “prefer not to say,” so these values need to be handled carefully during cleaning. Finally, this project mainly uses financial knowledge variables as predictors, so the models intentionally leave out many potentially important factors such as demographics, employment, household structure, and broader financial behavior.

## III. Modeling Approach

### Classification Targets

This project treats each financial vulnerability outcome as a separate binary classification problem. The five main targets are poverty status, fraud exposure, household income, savings, and income volatility. For each target, the outcome is coded so that `1` represents the financial condition of interest, such as being below a selected income threshold, reporting fraud exposure, having lower savings, or experiencing more income volatility. Each target is modeled separately using the same set of financial knowledge, numeracy, and self-assessed knowledge predictors.

### Models Compared

We compare several supervised classification models: logistic regression, LASSO logistic regression with cross-validation, decision tree, random forest, Bernoulli Naive Bayes, gradient boosting, and CatBoost. Logistic regression is used as the main interpretable baseline because its coefficients are easy to explain. LASSO logistic regression is included for variable selection and regularization. Decision tree, random forest, gradient boosting, and CatBoost are included to test whether nonlinear relationships improve predictive performance. Bernoulli Naive Bayes is included as a simple comparison model for the binary financial knowledge variables. CatBoost is included as an extension model because it is a strong gradient boosting method for tabular data, although it is not the main baseline model.

### Evaluation Metrics

Model performance is evaluated using accuracy, balanced accuracy, ROC AUC, precision, recall, F1 score, and confusion matrices. Because several target variables are imbalanced, accuracy alone can be misleading. For that reason, the main comparison emphasizes ROC AUC, balanced accuracy, recall, and F1 score. ROC AUC measures how well the model ranks positive cases above negative cases, while balanced accuracy accounts for performance on both classes. Precision, recall, and F1 score are used to evaluate how well each model identifies the positive class for each financial vulnerability target.

### Cross-Validation and Threshold Tuning

The data are split into a training set and a held-out test set. When threshold tuning is used, the classification threshold is selected using cross-validation within the training set only. After the threshold is chosen, the model is refit on the full training set and evaluated once on the test set. This avoids using the test set to make modeling decisions. The default threshold tuning metric is balanced accuracy, since it gives weight to both classes and is more appropriate when outcomes are imbalanced.

## IV. Analysis and Results
### Federal Poverty Status

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.46 | 0.6854 | 0.7647 | 0.6986 | 0.4086 | 0.7237 | 0.5223 |
| LASSO Logistic Regression with CV | 0.48 | 0.6898 | 0.7643 | 0.6969 | 0.4116 | 0.7105 | 0.5212 |
| Decision Tree | 0.42 | 0.6079 | 0.7162 | 0.6595 | 0.3499 | 0.7579 | 0.4788 |
| Random Forest | 0.52 | 0.6992 | 0.7629 | 0.6967 | 0.4195 | 0.6921 | 0.5223 |
| Bernoulli Naive Bayes | 0.15 | 0.6785 | 0.7533 | 0.6941 | 0.4020 | 0.7237 | 0.5169 |
| Gradient Boosting | 0.19 | 0.6523 | 0.7671 | 0.7040 | 0.3880 | 0.8026 | 0.5232 |
| CatBoost | 0.21 | 0.6798 | 0.7675 | 0.7085 | 0.4073 | 0.7632 | 0.5311 |

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Fraud Exposure

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.50 | 0.5451 | 0.5714 | 0.5747 | 0.3456 | 0.6445 | 0.4500 |
| LASSO Logistic Regression with CV | 0.49 | 0.5171 | 0.5717 | 0.5606 | 0.3318 | 0.6635 | 0.4423 |
| Decision Tree | 0.54 | 0.5492 | 0.5492 | 0.5459 | 0.3285 | 0.5379 | 0.4079 |
| Random Forest | 0.47 | 0.4952 | 0.5721 | 0.5544 | 0.3248 | 0.6943 | 0.4426 |
| Bernoulli Naive Bayes | 0.20 | 0.4774 | 0.5759 | 0.5581 | 0.3244 | 0.7488 | 0.4527 |
| Gradient Boosting | 0.27 | 0.5014 | 0.5669 | 0.5474 | 0.3217 | 0.6564 | 0.4318 |
| CatBoost | 0.26 | 0.4979 | 0.5669 | 0.5584 | 0.3274 | 0.7014 | 0.4465 |

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Household Income
#### High Household Income 

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.53 | 0.6610 | 0.7222 | 0.6600 | 0.6258 | 0.6480 | 0.6367 |
| LASSO Logistic Regression with CV | 0.53 | 0.6610 | 0.7222 | 0.6600 | 0.6258 | 0.6480 | 0.6367 |
| Decision Tree | 0.52 | 0.6648 | 0.6954 | 0.6613 | 0.6385 | 0.6194 | 0.6288 |
| Random Forest | 0.52 | 0.6648 | 0.7204 | 0.6614 | 0.6381 | 0.6207 | 0.6293 |
| Bernoulli Naive Bayes | 0.61 | 0.6529 | 0.7090 | 0.6522 | 0.6162 | 0.6439 | 0.6298 |
| Gradient Boosting | 0.49 | 0.6579 | 0.7143 | 0.6565 | 0.6237 | 0.6398 | 0.6316 |
| CatBoost | 0.51 | 0.6592 | 0.7146 | 0.6540 | 0.6382 | 0.5921 | 0.6143 |

![High Household Income Comparison](visuals/plots/high_HHI_comparison.png)

- ROC AUC visualization

![High HHI Income ROC AUC](visuals/plots/high_HHI_roc_auc.png)

- Confusion matrix for recommended/best model

![High HHI Income Confusion Matrix](visuals/plots/high_HHI_confusion_matrices/cm_lasso_logistic_regression_with_cv.png)

- Feature importances (maybe)

#### Low Household Income

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.45 | 0.6773 | 0.7572 | 0.6860 | 0.4603 | 0.7065 | 0.5575 |
| LASSO Logistic Regression with CV | 0.45 | 0.6773 | 0.7572 | 0.6860 | 0.4603 | 0.7065 | 0.5575 |
| Decision Tree | 0.52 | 0.7098 | 0.7247 | 0.6544 | 0.4959 | 0.5239 | 0.5095 |
| Random Forest | 0.49 | 0.6892 | 0.7613 | 0.6956 | 0.4732 | 0.7109 | 0.5682 |
| Bernoulli Naive Bayes | 0.18 | 0.6767 | 0.7483 | 0.6849 | 0.4596 | 0.7043 | 0.5562 |
| Gradient Boosting | 0.24 | 0.6623 | 0.7584 | 0.6923 | 0.4488 | 0.7630 | 0.5652 |
| CatBoost | 0.25 | 0.6748 | 0.7593 | 0.7005 | 0.4605 | 0.7609 | 0.5738 |

- ROC AUC visualization

![Low HHI Income ROC AUC](visuals/plots/low_HHI_roc_auc.png)

- Confusion matrix for recommended/best model

![Low HHI Income Confusion Matrix](visuals/plots/low_HHI_confusion_matrices/cm_random_forest.png)


- Feature importances (maybe)

### Savings

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.43 | 0.6303 | 0.7516 | 0.6840 | 0.4575 | 0.8333 | 0.5907 |
| LASSO Logistic Regression with CV | 0.43 | 0.6303 | 0.7516 | 0.6840 | 0.4575 | 0.8333 | 0.5907 |
| Decision Tree | 0.40 | 0.5694 | 0.7184 | 0.6505 | 0.4177 | 0.8762 | 0.5657 |
| Random Forest | 0.45 | 0.6326 | 0.7415 | 0.6819 | 0.4587 | 0.8190 | 0.5880 |
| Bernoulli Naive Bayes | 0.21 | 0.5877 | 0.7105 | 0.6501 | 0.4256 | 0.8238 | 0.5612 |
| Gradient Boosting | 0.26 | 0.6265 | 0.7462 | 0.6794 | 0.4542 | 0.8262 | 0.5861 |
| CatBoost | 0.26 | 0.6220 | 0.7427 | 0.6766 | 0.4508 | 0.8286 | 0.5839 |

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Financial Volatility

- Model comparison table

| Model | Threshold | Accuracy | ROC AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|---------|-------------------|-----------|--------|----|
| Baseline Logistic Regression | 0.51 | 0.6418 | 0.6654 | 0.6206 | 0.9578 | 0.6450 | 0.7709 |
| LASSO Logistic Regression with CV | 0.51 | 0.6210 | 0.6631 | 0.6318 | 0.9612 | 0.6193 | 0.7533 |
| Decision Tree | 0.48 | 0.6791 | 0.6235 | 0.6048 | 0.9533 | 0.6903 | 0.8008 |
| Random Forest | 0.52 | 0.6785 | 0.7039 | 0.6536 | 0.9628 | 0.6822 | 0.7986 |
| Bernoulli Naive Bayes | 0.95 | 0.5900 | 0.6722 | 0.6376 | 0.9642 | 0.5828 | 0.7265 |
| Gradient Boosting | 0.94 | 0.6304 | 0.6795 | 0.6100 | 0.9561 | 0.6335 | 0.7621 |
| CatBoost | 0.95 | 0.5616 | 0.6510 | 0.6000 | 0.9569 | 0.5558 | 0.7032 |

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)


## V. Recommended Model(s) and Conclusions

## VI. Modeling Limitations and Potential Extensions

## VII. Rerun Instructions
