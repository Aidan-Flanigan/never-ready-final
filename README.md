# Using Financial Knowledge Measures to Predict Household Financial Vulnerability
# I. Problem Statement and Motivation
This project attempts to explore whether financial knowledge measures can predict different forms of household financial vulnerability, including poverty status, fraud exposure, household income, savings, and income volatility. The motivation is that financial well-being is not only about income, but also about whether households can manage day-to-day finances, absorb shocks, and avoid harmful financial outcomes. The CFPB National Financial Well-Being Survey is well suited for this question because it includes measures of financial knowledge, financial skill, financial behavior, income, savings, safety nets, and financial experiences for a national sample of U.S. adults. In this project, we use financial knowledge, numeracy, and self-assessed knowledge variables to build classification models for several vulnerability outcomes, then compare whether interpretable models such as logistic regression perform similarly to more flexible machine learning models such as random forests and gradient boosting

## II. Data

### Data Source and Collection

This project uses the Consumer Financial Protection Bureau’s National Financial Well-Being Survey Public Use File. The survey was conducted online in English and Spanish between October and December 2016 and includes 6,394 completed responses from U.S. adults. The sample was drawn from GfK’s KnowledgePanel and was designed to represent the adult population of the 50 U.S. states and Washington, D.C., with an additional oversample of adults aged 62 and older. The dataset includes survey responses collected by the CFPB as well as pre-existing panel information on respondents, such as household income, poverty status, and demographic characteristics.

### Data Summary and Relevant Variables

The analysis focuses on whether financial knowledge measures can predict several household financial vulnerability outcomes. The main predictor variables include the Lusardi-Mitchell financial knowledge items (`FK1correct`, `FK2correct`, `FK3correct`), the Knoll-Houts financial knowledge items (`KH1correct` to `KH9correct`), objective numeracy items (`ON1correct`, `ON2correct`), and self-assessed financial knowledge (`SUBKNOWL1`). The target variables are constructed as binary classification outcomes, including poverty status, fraud exposure, household income, savings level, and income volatility. These outcomes capture different dimensions of financial vulnerability, from low income and low savings to unstable income and exposure to financial fraud.

### Limitations of Data

The data are cross-sectional, so the models should be interpreted as predictive rather than causal. The survey measures associations between financial knowledge and financial vulnerability, but it cannot prove that financial knowledge causes better or worse financial outcomes. Some variables are self-reported, which may introduce measurement error or recall bias, especially for sensitive topics such as income, savings, and fraud exposure. The dataset also includes nonsubstantive response codes such as refusal, “don’t know,” and “prefer not to say,” so these values need to be handled carefully during cleaning. Finally, this project mainly uses financial knowledge variables as predictors, so the models intentionally leave out many potentially important factors such as demographics, employment, household structure, and broader financial behavior.

## III. Modeling Approach
### Classification Targets

### Models Compared

### Evaluation Metrics

### Cross-Validation and Threshold Tuning

## IV. Analysis and Results
### Federal Poverty Status

- Model comparison table

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Fraud Exposure

- Model comparison table

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Household Income

- Model comparison table

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Savings

- Model comparison table

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)

### Financial Volatility

- Model comparison table

- ROC AUC visualization

- Confusion matrix for recommended/best model

- Feature importances (maybe)


## V. Recommended Model(s) and Conclusions

## VI. Modeling Limitations and Potential Extensions

## VII. Rerun Instructions