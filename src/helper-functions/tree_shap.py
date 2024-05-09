"""
The main script to build an explanatory model with specified encoded categorical variables.
"""
# Import modules
import numpy as np

class tree_cat_explainer:
    """
    Explanatory model derived from a decision tree or random forest model
    (regressor or classifier). Shap values are computed for feature groups.

    The cat_tree_shap function is heavily influenced by the tree_shap 
    function within the TreeExplainer class of the pytree.py script 
    located in shap/explainers/ of the SHAP module, particularly the 
    recurse, extend, unwind, and sum_unwind functions.

    https://github.com/shap/shap/blob/master/shap/explainers/pytree.py

    """
    def __init__(self, model, feature_groups = None):
        """
        :param model: tree-based model, random forest (scikit-learn).
        :type model: sklearn.tree.DecisionTreeRegressor;
        sklearn.tree.DecisionTreeClassifier;
        sklearn.ensemble.RandomForestRegressor;
        sklearn.ensemble.RandomForestClassifier.
        :param feature_groups: list of indices of the same group.
        :type feature_groups: list of lists (integer)
        """
        # Check model
        if str(type(model)) == "<class 'sklearn.tree._classes.DecisionTreeRegressor'>":
            self.rf = False
            self.classifier = False
        elif str(type(model)) == "<class 'sklearn.tree._classes.DecisionTreeClassifier'>":
            self.rf = False
            self.classifier = True
        elif str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestRegressor'>":
            self.rf = True
            self.classifier = False
        elif str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
            self.rf = True
            self.classifier = True
        else:
            raise TypeError("Model type not supported by tree_cat_explainer: " + str(type(model)))
        # Check if fitted
        if self.rf:
            try:
                self.trees = [model.estimators_[i].tree_ for i in range(model.n_estimators)]
            except:
                raise Exception("The model is not fitted")
        else:
            try:
                self.trees = [model.tree_]
            except:
                raise Exception("The model is not fitted")
        # Check if feature_groups is adapted

        self.n_features = self.trees[0].n_features
        if feature_groups == None:
            self.feature_groups = None
        else:
            if type(feature_groups) != list:
                raise TypeError("feature_groups must be a list, type(feature_groups): " + str(type(feature_groups)))
            feature_list = []
            # Check types
            self.feature_groups = []
            for index_group in feature_groups:
                if type(index_group) == int:
                    feature_list.append(index_group)
                    self.feature_groups.append([index_group])
                elif type(index_group) == list:
                    for ind in index_group:
                        if type(ind) == int:
                            feature_list.append(ind)
                        else:
                            raise TypeError("feature_groups must contain integers or lists of integers")
                    self.feature_groups.append(index_group)
                else:
                    raise TypeError("feature_groups must contain integers or lists of integers")
            # Check number of features
            feature_list = list(dict.fromkeys(feature_list))
            if len(feature_list) != self.n_features:
                raise Exception("Wrong number of features in feature_groups")
            if (max(feature_list) != self.n_features - 1) or (min(feature_list) != 0):
                raise Exception("Wrong feature indices in feature_groups")
        # Transform list of values (output of each leaf)
        self.trees_value = []
        for tree in self.trees:
            if self.classifier:
                # Normalise values to obtain probabilities for each leaf (classification)
                self.trees_value.append(tree.value[:,0]/tree.value[:,0].sum(axis = 1).reshape(tree.node_count,1))
            else :
                # Flatten vector of values for regression
                self.trees_value.append(tree.value.flatten())
    def shap_values(self, x_input):
        """
        Compute the shap values of an input array
        :param x: array of inputs of dimension (n,d) where n is the number of inputs
        and d the number of features.
        :type x: np.ndarray, list if a single input
        """
        if type(x_input) == list:
            try:
                x_tot = np.array(x_input)
            except:
                raise Exception("x_input must be a list or numpy.ndarray of integer")
        elif type(x_input) == np.ndarray:
            x_tot = x_input.copy()
        else:
            raise Exception("x_input must be a list or numpy.ndarray of integer")
        if len(x_tot.shape) == 1:
            x_tot = np.expand_dims(x_tot, axis=0)
        if x_tot.shape[1] != self.n_features:
            raise Exception("x_input have a wrong number of features")
        if self.feature_groups == None:
            if self.classifier:
                phi = np.zeros((x_tot.shape[0], x_tot.shape[1], self.trees_value[0].shape[1]))
            else:
                phi = np.zeros(x_tot.shape)
        else:
            if self.classifier:
                phi = np.zeros((x_tot.shape[0], len(self.feature_groups), self.trees_value[0].shape[1]))
            else:
                phi = np.zeros((x_tot.shape[0], len(self.feature_groups)))
        nb_trees = len(self.trees)
        for ind_tree, tree in enumerate(self.trees):
            for input_index, x in enumerate(x_tot):
                maxd = tree.max_depth + 2
                s = (maxd * (maxd + 1)) // 2
                node_features = tree.feature
                children_left = tree.children_left
                children_right = tree.children_right
                node_thresholds = tree.threshold
                node_fraction = tree.weighted_n_node_samples
                def extend(m, pz, po, pi, l):
                    """
                    """
                    m["feature"][l] = pi
                    m["z"][l] = pz
                    m["o"][l] = po
                    m["weight"][l] = int(l==0)
                    for i in range(l-1,-1,-1):
                        m["weight"][i+1] += po *  m["weight"][i] * (i + 1) / (l + 1)
                        m["weight"][i] = pz * m["weight"][i] * (l - i)/(l + 1)
                def unwind(m, i, l):
                    """
                    """
                    n = m["weight"][l]
                    i = int(i)
                    o = m["o"][i]
                    z = m["z"][i]
                    for j in range(l-1,-1,-1):
                        if o != 0:
                            t = m["weight"][j]
                            m["weight"][j] = n * (l + 1) / ((j + 1) * o)
                            n = t - m["weight"][j] * z * (l - j)/ (l + 1)
                        else:
                            m["weight"][j] = (m["weight"][j] * (l + 1)) / (z * (l - j))
                    for j in range(i, l):
                        m["feature"][j] = m["feature"][j+1]
                        m["z"][j] = m["z"][j+1]
                        m["o"][j] = m["o"][j+1]

                def sum_unwind(m, i, l):
                    o = m["o"][i]
                    z = m["z"][i]
                    n = m["weight"][l]
                    tot = 0
                    for j in range(l - 1, -1, -1):
                        if o != 0:
                            t = n * (l+1)/((j+1) * o)
                            tot += t
                            n = m["weight"][j] - t * z * (l - j) / (l + 1)
                        else:
                            tot += (m["weight"][j] / z) / ((l - j) / (l + 1))
                    return tot
                
                def recurse(j, node_path, pz, po, pi, l, parent):

                    """

                    """
                    node_path[f"node {j}"] = {}
                    for key in node_path[f"node {parent}"]:
                        node_path[f"node {j}"][key] = node_path[f"node {parent}"][key][l + 1:]
                        node_path[f"node {j}"][key][:l + 1] = node_path[f"node {parent}"][key][:l + 1]

                    extend(node_path[f"node {j}"], pz, po, pi, l)
                    left = children_left[j]
                    right = children_right[j]
                    if right < 0:
                        for i in range(1, l+1):
                            w = sum_unwind(node_path[f"node {j}"], i, l)
                            if self.feature_groups == None: 
                                feature_index = int(node_path[f"node {j}"]["feature"][i])
                                phi[input_index][feature_index] += (1 / nb_trees) * w * (node_path[f"node {j}"]["o"][i] - node_path[f"node {j}"]["z"][i]) * self.trees_value[ind_tree][j]
                            else:
                                for group_index, group in enumerate(self.feature_groups):
                                    if node_path[f"node {j}"]["feature"][i] in group:
                                        break
                                phi[input_index][group_index] += (1 / nb_trees) * w * (node_path[f"node {j}"]["o"][i] - node_path[f"node {j}"]["z"][i]) * self.trees_value[ind_tree][j]

                    else:
                        split = x[node_features[j]] <= node_thresholds[j]
                        h, c  = (left, right) * split + (right, left) * (1 - split)
                        iz, io = 1, 1

                        k = 0
                        if self.feature_groups == None:
                            while (k <= l): 
                                if node_path[f"node {j}"]["feature"][k] == node_features[j]:
                                    break
                                k += 1
                        else:
                            for group_index, group in enumerate(self.feature_groups):
                                if node_features[j] in group:
                                    break
                            while (k <= l):
                                if node_path[f"node {j}"]["feature"][k] in self.feature_groups[group_index]:
                                    break
                                k += 1
                        if k != l + 1:
                            iz, io = node_path[f"node {j}"]["z"][k], node_path[f"node {j}"]["o"][k]
                            unwind(node_path[f"node {j}"], k, l)
                            l -= 1

                        recurse(h, node_path, iz * node_fraction[h]/node_fraction[j], io, node_features[j], l + 1, j)
                        recurse(c, node_path, iz * node_fraction[c]/node_fraction[j], 0, node_features[j], l + 1, j)
                recurse(0, {"node -1": {"weight": np.zeros(s),
                            "z": np.zeros(s),
                            "o": np.zeros(s),
                            "feature": np.zeros(s)}}, 1, 1, -1, l = 0, parent = -1)
        if phi.shape[0] == 1:
            return phi[0]
        return phi