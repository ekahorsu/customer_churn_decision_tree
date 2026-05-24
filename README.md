# Decision Trees and the Bias-Variance Tradeoff: Diagnosing a Churn Classifier

A from-first-principles implementation of a decision tree classifier in NumPy, validated against scikit-learn on a realistic customer-churn dataset, with a full bias-variance diagnostic and a Random Forest comparison.

## What this project is

Decision trees are deceptively simple. They're easy to draw on a whiteboard, easy to explain to a stakeholder, and easy to overfit beyond recognition. This project treats a decision tree as both an algorithm to implement *and* a model whose behavior must be diagnosed:

1. **From-scratch implementation in NumPy**, entropy, information gain, recursive splitting, prediction
2. **Validation against scikit-learn**, to confirm the implementation is correct on real data
3. **Bias-variance diagnosis**, sweeping `max_depth` with 5-fold cross-validation to read off the classic bias-variance curve
4. **Ensembling comparison**, Random Forest vs. the single tree, to show what variance reduction looks like in practice

The dataset is a synthetic but realistic **customer churn** problem (2,000 customers, 12 features, 25.6% churn rate, 3% label noise) generated in-notebook with `sklearn.datasets.make_classification`. This keeps the notebook fully self-contained, no network access or external downloads required, while still posing a non-trivial supervised learning problem.

## What's inside

| Part | Topic |
|------|-------|
| 1 | Decision tree from scratch: entropy, information gain, recursive splitting |
| 2 | Real-world application to customer churn + scikit-learn validation |
| 3 | Bias-variance diagnosis via 5-fold cross-validation depth sweep |
| 4 | Random Forest comparison and the variance-reduction tradeoff |
| 5 | Reflection and lessons learned |

## Results

**From-scratch vs. scikit-learn (max_depth = 4):**
- Training accuracy: 0.7919 (both implementations, identical to 4 decimal places)
- Test accuracy: 0.7525 (both implementations, identical to 4 decimal places)

**Bias-variance sweep (5-fold CV over max_depth 1-15):**
- Best `max_depth` selected by cross-validation: 7
- Classic underfitting → sweet spot → overfitting curve clearly visible

**Random Forest (200 trees) vs. single best tree:**
- Forest improves CV accuracy by a few percentage points, the standard variance-reduction win
- Tradeoff: lost interpretability of the single tree

## How to run

```bash
git clone https://github.com/ekahorsu/customer_churn_decision_tree.git
cd customer_churn_decision_tree
pip install -r requirements.txt
jupyter notebook customer_churn_decision_tree.ipynb
```

Tested with Python 3.10. The dataset is generated inside the notebook, no external data files needed.

## Tech stack

- **NumPy**, all from-scratch math (entropy, information gain, recursive tree building)
- **pandas**, light data inspection
- **Matplotlib**, bias-variance plots and feature-importance bars
- **scikit-learn**, synthetic dataset generation, validation baselines, Random Forest, cross-validation
- **Jupyter**, notebook environment
