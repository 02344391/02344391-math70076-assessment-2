"""
In this script, we compare different approaches to calculating SHAP values for 
categorical data. The first part illustrates the difference in results with a 
very simple example. The second part calculates SHAP values for different dataset 
configurations for binary classification or regression.
"""

# Import modules and functions
import sys
import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
# Import scripts
sys.path.append(os.path.abspath("../src/helper-functions"))
import tree_shap
import create_categories
import plot_tree_path

###############################################################################
# Toy example
###############################################################################
"""
A very simple decision tree to exhibit a difference according to the way of 
calculating the SHAP values.
"""
# Construct simple dataset 
target = [-10] * 10 + [10] * 5 + [5] * 20 + [14] * 5
cat_1 = [1] * 10 + [0] * 30
x = [60] * 5 + [45] * 30 +  [50] * 5
cat_2 = [1] * 15 + [0] * 25
data = pd.DataFrame({"cat_1": cat_1,
                     "cat_2": cat_2,
                     "x": x,
                     "target": target})
# Build model
Tree = DecisionTreeRegressor()
Tree.fit(data.values[:,:-1], data.values[:,-1])

input_shap = [0,1,60] # Analyse input
# Plot tree path of input knowing 0, 1, 2 and 3 features
## 0 known feature
fig, ax = plt.subplots(1, figsize = (4,5))
plot_tree_path.plot_tree_color(ax, Tree, input_shap, data.columns[:-1], [])
plt.savefig(os.path.abspath("../outputs/plot_tree/figures/path_0_known.pdf"))
## 1 known feature
fig, ax = plt.subplots(1,3, figsize = (12,5))
plot_tree_path.plot_tree_color(ax[0], Tree, input_shap, data.columns[:-1], [0])
plot_tree_path.plot_tree_color(ax[1], Tree, input_shap, data.columns[:-1], [1])
plot_tree_path.plot_tree_color(ax[2], Tree, input_shap, data.columns[:-1], [2])
# plot_tree_path.plot_tree_color(ax, Tree, input_shap, data.columns[:-1], [0,1])
plt.savefig(os.path.abspath("../outputs/plot_tree/figures/path_1_known.pdf"))
## 2 known features
fig, ax = plt.subplots(1,3, figsize = (12,5))
plot_tree_path.plot_tree_color(ax[0], Tree, input_shap, data.columns[:-1], [0,1])
plot_tree_path.plot_tree_color(ax[1], Tree, input_shap, data.columns[:-1], [1,2])
plot_tree_path.plot_tree_color(ax[2], Tree, input_shap, data.columns[:-1], [0,2])
plt.savefig(os.path.abspath("../outputs/plot_tree/figures/path_2_known.pdf"))

## 3 known features
fig, ax = plt.subplots(1, figsize = (4,5))
plot_tree_path.plot_tree_color(ax, Tree, input_shap, data.columns[:-1], [0,1,2])
plt.savefig(os.path.abspath("../outputs/plot_tree/figures/path_3_known.pdf"))

###############################################################################
# Build datasets and compare both methods of computing SHAP values
###############################################################################

# Create dictionary of configuration
"""
All datasets consist of 300 samples, the test/train ratio is 1/3.
For 5 informative features, total 7 features, we test 20 different datasets
with 1, 3, 5 categorical features with 4 categories each.
Classification and regression.
"""

seed = 2344391
path_folder = "../outputs/comparison-shap/data/"

for nb_var_cat in [1, 3, 5]:
    path_pkl = os.path.abspath(path_folder + f"nb_var_cat_{nb_var_cat}.pkl")
    # Check if file already exists
    if os.path.exists(path_pkl):
        continue
    # Dictionary of results
    results = {}
    if nb_var_cat == 1:
        feature_groups = [i for i in range(6)] + [[6, 7, 8]]
    if nb_var_cat == 3:
        feature_groups = [i for i in range(4)] + [list(np.array([4, 5, 6]) + i * 3) for i in range(3)]
    if nb_var_cat == 5:
        feature_groups = [0, 1] + [list(np.array([2, 3, 4]) + i * 3) for i in range(5)]
    for iteration in range(20):
        results[f"iteration {iteration}"] = {}
        # Classification
        X_classif, y_classif = make_classification(n_samples = 300, 
                                                   n_features = 7,
                                                   n_informative = 5, 
                                                   n_redundant = 0,
                                                   random_state = seed + iteration, 
                                                   shuffle = False)
        X_classif_df = pd.DataFrame(X_classif, columns= ["X_" + str(i) for i in range(X_classif.shape[1])])
        for var_cat in range(nb_var_cat):
            X_classif_df[f"X_{var_cat}"] = create_categories.transform_num_to_cat(X = X_classif[:, var_cat], 
                                                                                  feature_name = f"Classif_{var_cat}", 
                                                                                  nb_cat = 4,
                                                                                  randomness = 0.05)
        X_classif_cat = X_classif_df.copy()
        X_classif_cat = pd.get_dummies(X_classif_cat, 
                               columns= [f"X_{var_cat}" for var_cat in range(nb_var_cat)], 
                               drop_first= True)
        X_classif_train, X_classif_test, y_classif_train, y_classif_test = train_test_split(X_classif_cat, 
                                                                                            y_classif, 
                                                                                            test_size=0.2, 
                                                                                            random_state=42)
        classifier = RandomForestClassifier(n_estimators=20, # limit the number of estimators (slow algorithm)
                                            random_state= seed + iteration)
        classifier.fit(X_classif_train, y_classif_train)
        results[f"iteration {iteration}"]["X_classif_train"] = X_classif_train.copy()
        results[f"iteration {iteration}"]["X_classif_test"] = X_classif_test.copy()
        # Explainer Shap (classification)
        explain_classif_shap = shap.TreeExplainer(classifier)
        shap_classif_train = explain_classif_shap.shap_values(X_classif_train)
        shap_classif_test = explain_classif_shap.shap_values(X_classif_test)
        results[f"iteration {iteration}"]["shap_classif_train"] = tree_shap.sum_cat_shap(shap_classif_train, 
                                                                                        feature_groups, 
                                                                                        n_classes=2)[:,:,0]
        results[f"iteration {iteration}"]["shap_classif_test"] = tree_shap.sum_cat_shap(shap_classif_test, 
                                                                                         feature_groups, 
                                                                                         n_classes=2)[:,:,0]
        # Explainer Shap Cat (classification)
        explain_classif_cat_shap = tree_shap.tree_cat_explainer(classifier, 
                                                                feature_groups = feature_groups)
        results[f"iteration {iteration}"]["shap_cat_classif_train"] = explain_classif_cat_shap.shap_values(X_classif_train)[:,:,0]
        results[f"iteration {iteration}"]["shap_cat_classif_test"] = explain_classif_cat_shap.shap_values(X_classif_test)[:,:,0]
        # Regression
        X_reg, y_reg = make_regression(n_samples=300, 
                                       n_features=7,
                                       n_informative=5,
                                       bias = 0,
                                       noise = 0.1,
                                       random_state = seed + iteration, 
                                       shuffle=False)
        X_reg_df = pd.DataFrame(X_reg, columns= ["X_" + str(i) for i in range(X_reg.shape[1])])
        for var_cat in range(nb_var_cat):
            X_reg_df[f"X_{var_cat}"] = create_categories.transform_num_to_cat(X = X_reg[:, var_cat], 
                                                                                  feature_name = f"Reg_{var_cat}", 
                                                                                  nb_cat = 4,
                                                                                  randomness = 0.05)
        X_reg_cat = X_reg_df.copy()
        X_reg_cat = pd.get_dummies(X_reg_cat, 
                               columns= [f"X_{var_cat}" for var_cat in range(nb_var_cat)], 
                               drop_first= True)
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_cat, y_reg, test_size=0.2, random_state = seed + iteration)
        regressor = RandomForestRegressor(n_estimators = 20, # limit the number of estimators (slow algorithm)
                                          random_state= seed + iteration)
        regressor.fit(X_reg_train, y_reg_train)
        results[f"iteration {iteration}"]["X_reg_train"] = X_reg_train.copy()
        results[f"iteration {iteration}"]["X_reg_test"] = X_reg_test.copy()
        # Explainer Shap (regression)
        explain_reg_shap = shap.TreeExplainer(regressor)
        shap_reg_train = explain_reg_shap.shap_values(X_reg_train)
        shap_reg_test = explain_reg_shap.shap_values(X_reg_test)
        results[f"iteration {iteration}"]["shap_reg_train"] = tree_shap.sum_cat_shap(shap_reg_train, 
                                                                                     feature_groups)
        results[f"iteration {iteration}"]["shap_reg_test"] = tree_shap.sum_cat_shap(shap_reg_test, 
                                                                                    feature_groups)
        # Explainer Shap Cat (regression)
        explain_reg_cat_shap = tree_shap.tree_cat_explainer(regressor, 
                                                            feature_groups = feature_groups)
        results[f"iteration {iteration}"]["shap_cat_reg_train"] = explain_reg_cat_shap.shap_values(X_reg_train)
        results[f"iteration {iteration}"]["shap_cat_reg_test"] = explain_reg_cat_shap.shap_values(X_reg_test)
        # Add scores
        results[f"iteration {iteration}"]["accuracy"] = accuracy_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["precision"] = precision_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["recall"] = recall_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["mae"] = mean_absolute_error(y_reg_test, regressor.predict(X_reg_test))
    # Save results
    with open(path_pkl, 'wb') as f:
        pickle.dump(results, f)


