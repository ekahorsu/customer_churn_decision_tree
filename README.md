# Decision Trees and the Bias-Variance Tradeoff: Credit Risk Classification

A decision tree classifier implemented from first principles in NumPy and applied to a real credit-scoring problem, validated against scikit-learn, with a cross-validated bias-variance diagnosis and a random forest comparison.

## Overview

This project predicts whether a loan applicant represents a good or bad credit risk using the German Credit dataset (1,000 real applicants, 20 features). It combines a from-scratch implementation of the decision tree algorithm with the model-diagnosis workflow used in practice:

1. Implement entropy, information gain, and recursive binary splitting in NumPy, then validate against scikit-learn.
2. Diagnose the bias-variance tradeoff by sweeping tree depth with cross-validation.
3. Compare the major model families, progressing from a single tree to a random forest (bagging) and gradient boosting (XGBoost).

The dataset mixes categorical features (credit history, purpose, savings, employment) with numeric ones (loan amount, duration, age), which makes it a realistic example of the mixed-type tabular data common in financial applications. The data ships with the repository, so the notebook runs without any network access.

## Selected results

- The from-scratch tree and scikit-learn's `DecisionTreeClassifier` produce identical accuracy (0.765 train, 0.685 test) at matched hyperparameters, confirming the implementation is correct.
- The most influential features are checking-account status, loan duration, loan amount, and savings, consistent with established credit-risk modeling.
- Cross-validation selects a tree depth of 5 as the best balance between underfitting and overfitting.
- Across the three model families, performance improves at each step:

| Model | Test accuracy | Recall (bad-risk class) |
|-------|--------------|------------------------|
| Single tree (best depth) | 0.70 | 0.18 |
| Random forest (bagging) | 0.74 | 0.43 |
| XGBoost (boosting) | 0.75 | 0.52 |

Accuracy in this range is expected: credit risk is intrinsically difficult to predict from these features, and the results are in line with the known benchmark performance for this dataset. Recall on the bad-risk class is treated as the more informative metric, since the cost of approving a loan that defaults is higher than the cost of declining one that would have been repaid. The progression from a single tree to bagging to boosting raises that recall from 0.18 to 0.52, which is the most meaningful improvement for this application.

## Challenges and Solutions

### Encoding mixed-type features

**Challenge.** The dataset combines categorical attributes such as credit history, purpose, and savings with numeric ones such as loan amount and age. A decision tree splits on ordered thresholds, so the categorical columns could not be used in their raw string form.

**Solution.** The categorical columns were converted to integer codes with an ordinal encoder, allowing the tree to treat them as ordered values and search for thresholds among them. This is a deliberate simplification: it imposes an ordering on categories that may not have a natural one. Dedicated categorical handling, as found in XGBoost and LightGBM, would instead consider arbitrary groupings, and this is noted as a limitation of the from-scratch approach.

### Choosing the right evaluation metric under class imbalance

**Challenge.** The target is imbalanced at roughly 70 percent good and 30 percent bad risk. With this distribution, overall accuracy is a misleading measure of quality, since a model could score reasonably well simply by favoring the majority class while failing to identify the applicants who matter most.

**Solution.** Recall on the bad-risk class was adopted as the primary metric, reflecting the asymmetric cost of errors in lending: approving a loan that later defaults is far more costly than declining one that would have been repaid. Tracking this metric across the three models showed it rising from 0.18 to 0.52, a more meaningful gain than the change in overall accuracy.

### Selecting model complexity without overfitting

**Challenge.** A single train/test accuracy figure does not reveal whether a model is too simple or too complex, which makes the choice of tree depth difficult to justify on its own.

**Solution.** Tree depth was swept across a range of values, recording both training accuracy and five-fold cross-validation accuracy at each depth. The resulting curves exposed the bias-variance tradeoff directly, with cross-validation identifying depth five as the best balance. Crucially, the depth was selected without consulting the test set, which was reserved for a single final evaluation.

## How to run

```bash
git clone https://github.com/ekahorsu/decision-tree-credit-risk.git
cd decision-tree-credit-risk
pip install -r requirements.txt
jupyter notebook credit_risk_decision_tree.ipynb
```

Tested with Python 3.10. No internet connection is required; the dataset is included.

## Tech stack

- **NumPy** for the from-scratch tree implementation
- **pandas** for data loading and preprocessing
- **Matplotlib** for the bias-variance and feature-importance plots
- **scikit-learn** for validation baselines, cross-validation, and the random forest
- **XGBoost** for gradient boosting

## Repository contents

- `credit_risk_decision_tree.ipynb` - the full analysis notebook
- `german_credit.csv` - the dataset (1,000 applicants, 20 features)
- `requirements.txt` - dependencies

## Data source

The dataset is the Statlog (German Credit Data) set from the UCI Machine Learning Repository, originally donated by Professor Hans Hofmann of the University of Hamburg. It contains 1,000 loan applicants described by 20 attributes and a binary credit-risk label.

Original source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The copy included in this repository was obtained from a public mirror of the dataset and is provided for convenience and reproducibility, so that the notebook runs without any network access.
