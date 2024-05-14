"""
In this script, we compare different approaches to calculating SHAP values for 
categorical data. The first part illustrates the difference in results with a 
very simple example. The second part calculates SHAP values for different dataset 
configurations for binary classification or regression. The results or saved in
outputs/comparison-shap/data because the algorithm is slow: tree_cat_explainer 
is not optimised (coded in Python instead of C++).
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error as mae
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
    # Construction of feature_groups
    if nb_var_cat == 1:
        feature_groups = [i for i in range(6)] + [[6, 7, 8]]
    if nb_var_cat == 3:
        feature_groups = [i for i in range(4)] + [list(np.array([4, 5, 6]) + i * 3) for i in range(3)]
    if nb_var_cat == 5:
        feature_groups = [0, 1] + [list(np.array([2, 3, 4]) + i * 3) for i in range(5)]
    # Construct 20 datasets for classification and regression
    for iteration in range(20):
        results[f"iteration {iteration}"] = {}
        # Classification
        X_classif, y_classif = make_classification(n_samples = 300, 
                                                   n_features = 7,
                                                   n_informative = 5, 
                                                   n_redundant = 0,
                                                   random_state = seed + iteration, 
                                                   shuffle = False)
        # Construct dataframe
        X_classif_df = pd.DataFrame(X_classif, columns= ["X_" + str(i) for i in range(X_classif.shape[1])])
        # Transform numerical data to categorical data
        for var_cat in range(nb_var_cat):
            X_classif_df[f"X_{var_cat}"] = create_categories.transform_num_to_cat(X = X_classif[:, var_cat], 
                                                                                  feature_name = f"Classif_{var_cat}", 
                                                                                  nb_cat = 4,
                                                                                  randomness = 0.05)
        # Copy dataframe to encode categorical features
        X_classif_cat = X_classif_df.copy()
        X_classif_cat = pd.get_dummies(X_classif_cat, 
                               columns= [f"X_{var_cat}" for var_cat in range(nb_var_cat)], 
                               drop_first= True)
        # Split into train/test sets
        X_classif_train, X_classif_test, y_classif_train, y_classif_test = train_test_split(X_classif_cat, 
                                                                                            y_classif, 
                                                                                            test_size=0.2, 
                                                                                            random_state=42) # Wrong seed, realised after.
                                                                                                            # but it should not be a problem
                                                                                                            # since make_classification has
                                                                                                            # a changing seed
        # Construct model
        classifier = RandomForestClassifier(n_estimators=20, # limit the number of estimators (slow algorithm)
                                            random_state= seed + iteration)
        classifier.fit(X_classif_train, y_classif_train)
        # Update result: X_train, X_test
        results[f"iteration {iteration}"]["X_classif_train"] = X_classif_train.copy()
        results[f"iteration {iteration}"]["X_classif_test"] = X_classif_test.copy()
        # Explainer Shap (classification)
        explain_classif_shap = shap.TreeExplainer(classifier)
        shap_classif_train = explain_classif_shap.shap_values(X_classif_train)
        shap_classif_test = explain_classif_shap.shap_values(X_classif_test)
        # Update results shap
        results[f"iteration {iteration}"]["shap_classif_train"] = tree_shap.sum_cat_shap(shap_classif_train, 
                                                                                        feature_groups, 
                                                                                        n_classes=2)[:,:,0]
        results[f"iteration {iteration}"]["shap_classif_test"] = tree_shap.sum_cat_shap(shap_classif_test, 
                                                                                         feature_groups, 
                                                                                         n_classes=2)[:,:,0]
        # Explainer Shap Cat (classification)
        explain_classif_cat_shap = tree_shap.tree_cat_explainer(classifier, 
                                                                feature_groups = feature_groups)
        # Update results shap cat
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
        # Dataframe
        X_reg_df = pd.DataFrame(X_reg, columns= ["X_" + str(i) for i in range(X_reg.shape[1])])
        # Create categorical features
        for var_cat in range(nb_var_cat):
            X_reg_df[f"X_{var_cat}"] = create_categories.transform_num_to_cat(X = X_reg[:, var_cat], 
                                                                                  feature_name = f"Reg_{var_cat}", 
                                                                                  nb_cat = 4,
                                                                                  randomness = 0.05)
        # Copy dataframe to encode categorical features
        X_reg_cat = X_reg_df.copy()
        X_reg_cat = pd.get_dummies(X_reg_cat, 
                               columns= [f"X_{var_cat}" for var_cat in range(nb_var_cat)], 
                               drop_first= True)
        # Train/test split
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_cat, 
                                                                            y_reg, test_size=0.2, 
                                                                            random_state = seed + iteration)
        # Construct model
        regressor = RandomForestRegressor(n_estimators = 20, # limit the number of estimators (slow algorithm)
                                          random_state= seed + iteration)
        regressor.fit(X_reg_train, y_reg_train)
        # Update result: X_train, X_test
        results[f"iteration {iteration}"]["X_reg_train"] = X_reg_train.copy()
        results[f"iteration {iteration}"]["X_reg_test"] = X_reg_test.copy()
        # Explainer Shap (regression)
        explain_reg_shap = shap.TreeExplainer(regressor)
        shap_reg_train = explain_reg_shap.shap_values(X_reg_train)
        shap_reg_test = explain_reg_shap.shap_values(X_reg_test)
        # Update results shap
        results[f"iteration {iteration}"]["shap_reg_train"] = tree_shap.sum_cat_shap(shap_reg_train, 
                                                                                     feature_groups)
        results[f"iteration {iteration}"]["shap_reg_test"] = tree_shap.sum_cat_shap(shap_reg_test, 
                                                                                    feature_groups)
        # Explainer Shap Cat (regression)
        explain_reg_cat_shap = tree_shap.tree_cat_explainer(regressor, 
                                                            feature_groups = feature_groups)
        # Update results shap cat
        results[f"iteration {iteration}"]["shap_cat_reg_train"] = explain_reg_cat_shap.shap_values(X_reg_train)
        results[f"iteration {iteration}"]["shap_cat_reg_test"] = explain_reg_cat_shap.shap_values(X_reg_test)
        # Add scores
        results[f"iteration {iteration}"]["accuracy"] = accuracy_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["precision"] = precision_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["recall"] = recall_score(y_classif_test, classifier.predict(X_classif_test))
        results[f"iteration {iteration}"]["mae"] = mae(y_reg_test, regressor.predict(X_reg_test))
    # Save results: slow algorithm
    with open(path_pkl, 'wb') as f:
        pickle.dump(results, f)
# Load data
with open(os.path.abspath(path_folder + f"nb_var_cat_{1}.pkl"), 'rb') as f:
    results_1 = pickle.load(f)
with open(os.path.abspath(path_folder + f"nb_var_cat_{3}.pkl"), 'rb') as f:
    results_3 = pickle.load(f)
with open(os.path.abspath(path_folder + f"nb_var_cat_{5}.pkl"), 'rb') as f:
    results_5 = pickle.load(f)

"""
Print mean results on the test sets over all the datasets for each quantity of categorical feature.
Boxplots of differences of absolute values.
Ranking:
For the ranking of the SHAP values, we decide to set to 0 all SHAP values contributing less than 10% 
to the prediction. Then, we examine the differences in ranking of the SHAP values for the most 
significant variables, which seems more pertinent.
"""
fig, ax_boxplot = plt.subplots(3,2, figsize = (10,12))
for order, res in enumerate([(results_1, 1), (results_3, 3), (results_5, 5)]):
    print(f"Accuracy {res[1]} cat: ")
    print(np.array([res[0][f"iteration {iteration}"]["accuracy"] for iteration in range(20)]).mean())
    print(f"Precision {res[1]} cat: ")
    print(np.array([res[0][f"iteration {iteration}"]["precision"] for iteration in range(20)]).mean())
    print(f"Recall {res[1]} cat: ")
    print(np.array([res[0][f"iteration {iteration}"]["recall"] for iteration in range(20)]).mean())
    print(f"MAE {res[1]} cat:")
    print(np.array([res[0][f"iteration {iteration}"]["mae"] for iteration in range(20)]).mean())
    # Update Boxplots
    boxplot_array_train = np.zeros((240,20))
    boxplot_array_test = np.zeros((60,20))
    diff_ranking_count_train = 0
    diff_ranking_count_test = 0
    for iteration in range(20):
        # Train set
        norm_abs_shap_cat_train = tree_shap.normalise_absolute_shap_value(res[0][f"iteration {iteration}"]["shap_cat_classif_train"])
        norm_abs_shap_train = tree_shap.normalise_absolute_shap_value(res[0][f"iteration {iteration}"]["shap_classif_train"])
        ## Ranking of abs shap values
        # Set to zero non significant shap values
        condition_10_percent_cat_train = [norm_abs_shap_cat_train > 0.1]
        choice_10_percent_cat_train = [norm_abs_shap_cat_train]
        significant_shap_cat_train = np.select(condition_10_percent_cat_train,
                                          choice_10_percent_cat_train,
                                          0)
        condition_10_percent_train = [norm_abs_shap_train > 0.1]
        choice_10_percent_train = [norm_abs_shap_train]
        significant_shap_train = np.select(condition_10_percent_train,
                                          choice_10_percent_train,
                                          0)
        order_shap_cat_train = np.argsort(significant_shap_cat_train, axis = 1)
        order_shap_train = np.argsort(significant_shap_train, axis = 1)
        diff_ranking_count_train += (abs(order_shap_cat_train - order_shap_train).max(axis = 1) != 0).sum()
        # Test set
        norm_abs_shap_cat_test = tree_shap.normalise_absolute_shap_value(res[0][f"iteration {iteration}"]["shap_cat_classif_test"])
        norm_abs_shap_test = tree_shap.normalise_absolute_shap_value(res[0][f"iteration {iteration}"]["shap_classif_test"])
        ## Ranking
        # set to zero non significant shap values
        condition_10_percent_cat_test = [norm_abs_shap_cat_test > 0.1]
        choice_10_percent_cat_test = [norm_abs_shap_cat_test]
        significant_shap_cat_test = np.select(condition_10_percent_cat_test,
                                          choice_10_percent_cat_test,
                                          0)
        condition_10_percent_test = [norm_abs_shap_test > 0.1]
        choice_10_percent_test = [norm_abs_shap_test]
        significant_shap_test = np.select(condition_10_percent_test,
                                          choice_10_percent_test,
                                          0)
        order_shap_cat_test = np.argsort(significant_shap_cat_test, axis = 1)
        order_shap_test = np.argsort(significant_shap_test, axis = 1)
        diff_ranking_count_test += (abs(order_shap_cat_test - order_shap_test).max(axis = 1) != 0).sum()
        # Update boxplot
        boxplot_array_train[:, iteration] = abs(norm_abs_shap_cat_train - norm_abs_shap_train).mean(axis = 1)
        boxplot_array_test[:, iteration] = abs(norm_abs_shap_cat_test - norm_abs_shap_test).mean(axis = 1) 
    print("Difference in the ranking of the absolute values of the SHAP values when they are greater than 10%:")
    print(f"- Trainning set {res[1]} categories (error rate): {diff_ranking_count_train/(20*240)}")
    print(f"- Test set {res[1]} categories (error rate): {diff_ranking_count_test/(20*60)}")
    ax_boxplot[order, 0].boxplot(boxplot_array_train)
    ax_boxplot[order, 0].set_title(f"Differences of absolute values of SHAP values\n ({res[1]} categorical feature(s))")
    ax_boxplot[order, 0].set_xlabel("Training datasets")
    ax_boxplot[order, 1].boxplot(boxplot_array_train)
    ax_boxplot[order, 1].set_title(f"Differences of absolute values of SHAP values\n ({res[1]} categorical feature(s))")
    ax_boxplot[order, 1].set_xlabel("Test datasets")
plt.tight_layout()
plt.savefig(os.path.abspath("../outputs/comparison-shap/figures/boxplots.pdf"))
# Print
"""
Accuracy 1 cat: 
0.8241666666666667
Precision 1 cat: 
0.8558768677970567
Recall 1 cat: 
0.8077972262952102
MAE 1 cat:
43.99385788688136
Difference in the ranking of the absolute values of the SHAP values when they are greater than 10%:
- Trainning set 1 categories (error rate): 0.04541666666666667
- Test set 1 categories (error rate): 0.05
Accuracy 3 cat: 
0.8091666666666667
Precision 3 cat: 
0.8414943037151741
Recall 3 cat: 
0.790259958455523
MAE 3 cat:
57.726867659583434
Difference in the ranking of the absolute values of the SHAP values when they are greater than 10%:
- Trainning set 3 categories (error rate): 0.15083333333333335
- Test set 3 categories (error rate): 0.16666666666666666
Accuracy 5 cat: 
0.7475000000000002
Precision 5 cat:
0.796528188201923
Recall 5 cat:
0.7090450879765395
MAE 5 cat:
72.99489901991244
Difference in the ranking of the absolute values of the SHAP values when they are greater than 10%:
- Trainning set 5 categories (error rate): 0.2529166666666667
- Test set 5 categories (error rate): 0.24916666666666668
"""