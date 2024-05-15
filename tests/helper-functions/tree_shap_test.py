### Test treeshap
"""
Multiple tests to make sure that the shap values are computed well
"""

## Import libraries and functions
import sys
import os
import pandas as pd
import numpy as np
import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

## Import script 
sys.path.append(os.path.abspath("../../src/helper-functions"))
import tree_shap

## Test tree_cat_explainer
# Create dataset with categorical data for regression
target = [-10] * 10 + [10] * 5 + [5] * 20 + [14] * 5
cat_1 = [1] * 10 + [0] * 30
cat_2 = [1] * 15 + [0] * 25
x = [60] * 5 + [45] * 30 +  [50] * 5
data = pd.DataFrame({"cat_1": cat_1,
                     "cat_2": cat_2,
                     "x": x,
                     "target": target})
input_reg = np.array([[0,1,60], [1,0,40]])
# Fit model
reg_tree = DecisionTreeRegressor().fit(data.values[:, :-1], data.values[:, -1])
# First without groupings, comparison with shap module
np.testing.assert_allclose(tree_shap.tree_cat_explainer(reg_tree).shap_values(input_reg),
                           shap.TreeExplainer(reg_tree).shap_values(input_reg),
                           rtol = 1e-12)
print("ok regression")
# Then with groupings
true_shap_group = np.array([[  41/6,   25/6],
                            [-12.5       ,  -0.5       ]])
np.testing.assert_allclose(tree_shap.tree_cat_explainer(reg_tree, feature_groups= [[0,1], 2]).shap_values(input_reg),
                           true_shap_group, 
                           rtol = 1e-12)
print("ok regression groups")
# Create dataset for classification
X, y = make_classification(n_samples=100, n_features=5,
                           n_informative=3,
                           random_state=2344391, shuffle=False)
rf_classif = RandomForestClassifier(random_state=2344391)
rf_classif.fit(X, y)
input_classif = X[39]

np.testing.assert_allclose(tree_shap.tree_cat_explainer(rf_classif).shap_values(input_classif),
                           shap.TreeExplainer(rf_classif).shap_values(input_classif),
                           rtol = 1e-12)
print("ok classification")
# Test sum_cat_shap
np.testing.assert_allclose(tree_shap.sum_cat_shap(tree_shap.tree_cat_explainer(reg_tree).shap_values(input_reg),
                           feature_groups = [[0,1], 2]),
                           np.array([[191/36 + 25/18, 155/36],
                                     [-583/48 - 1/3,  -25/48]]))
print("ok sum")

# Test normalise_absolute_shap_value(shap_values, n_decimals = 3)
np.testing.assert_allclose(tree_shap.normalise_absolute_shap_value(tree_shap.tree_cat_explainer(reg_tree).shap_values(input_reg),
                           n_decimals = 4),
                           np.array([[0.4823, 0.1263, 0.3914],
                                     [0.9343, 0.0256, 0.0401]]))
print("ok normalise")