"""
The main script to build an explanatory model with specified encoded categorical variables.
"""
# Import modules
import numpy as np
# Main class
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
                        if (type(ind) == int) or (type(ind) == np.int32):
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
        :param x_input: array of inputs of dimension (n,d) where n is the number of inputs
        and d the number of features.
        :type x_input: pd.DataFrame, np.ndarray or list (only for a single input).
        :returns: array of Shap values for each feature, each observation.
        :rtype: np.ndarray.
        """
        if str(type(x_input)) == "<class 'pandas.core.frame.DataFrame'>":
            x_tot = x_input.values
        elif type(x_input) == list:
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
                tot_length = (maxd * (maxd + 1)) // 2
                node_features = tree.feature
                children_left = tree.children_left
                children_right = tree.children_right
                node_thresholds = tree.threshold
                node_fraction = tree.weighted_n_node_samples
                # Tree shap algorithm 
                # Consistent Individualized Feature Attribution for Tree Ensembles
                # page 4
                # Author: Scott M. Lundberg, Gabriel G. Erion, and Su-In Lee
                def extend(m, pz, po, pi, l):
                    """
                    Grow subsets of features from a given node according to a given fraction
                    of ones and zeros.
                    :param m: unique path of a given node
                    :type m: dicts with keys 'weigth','zero','one','feature'
                    :param pz: proportion of "zero" paths (where the feature is not in the considered set of features)
                    :type pz: float
                    :param po: proportion of "one" paths (where the feature is in the considered set of features)
                    :type po: float
                    :param pi: index of the parent split feature 
                    :type pi: int
                    :param l: current depth of the node
                    :type l: int
                    """
                    m["feature"][l] = pi
                    m["zero"][l] = pz
                    m["one"][l] = po
                    m["weight"][l] = int(l==0)
                    for i in range(l-1,-1,-1):
                        m["weight"][i+1] += po *  m["weight"][i] * (i + 1) / (l + 1)
                        m["weight"][i] = pz * m["weight"][i] * (l - i)/(l + 1)
                def unwind(m, i, l):
                    """
                    Reverse extend procedure when algorithm splits on the same feature twice
                    :param m: unique path of a given node
                    :type m: dicts with keys 'weigth','zero','one','feature'
                    :param i: depth of the node where to undo extension
                    :type i: int 
                    :param l: current depth of the algorithm
                    :type l: int
                    """
                    n = m["weight"][l]
                    i = int(i)
                    o = m["one"][i]
                    z = m["zero"][i]
                    for j in range(l-1,-1,-1):
                        if o != 0:
                            t = m["weight"][j]
                            m["weight"][j] = n * (l + 1) / ((j + 1) * o)
                            n = t - m["weight"][j] * z * (l - j)/ (l + 1)
                        else:
                            m["weight"][j] = (m["weight"][j] * (l + 1)) / (z * (l - j))
                    for j in range(i, l):
                        m["feature"][j] = m["feature"][j+1]
                        m["zero"][j] = m["zero"][j+1]
                        m["one"][j] = m["one"][j+1]
                def sum_unwind(m, i, l):
                    """
                    Undo each extension of the path inside a leaf to compute weights for each
                    feature in the path
                    :param m: unique path of a given node
                    :type m: dicts with keys 'weigth','zero','one','feature'
                    :param i: depth of the node where to undo extension
                    :type i: int 
                    :param l: current depth of the algorithm
                    :type l: int
                    """
                    o = m["one"][i]
                    z = m["zero"][i]
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
                    Update shap values when algorithm encouters a leaf, or update weights given the features
                    when it is an internal node.
                    :param j: current node the algorithm passes through
                    :type j: integer
                    :param node_path: dictionary of all the previous unique paths for every node
                    :type node_path: dict of dicts with keys 'weigth','zero','one','feature'
                    :param pz: proportion of "zero" paths (where the feature is not in the considered set)
                    :type pz: float
                    :param po: proportion of "one" paths (where the feature is in the considered set)
                    :type po: float
                    :param pi: index of the parent split feature 
                    :type pi: integer
                    :param l: current depth of the algorithm (of the node the algorithm passes through)
                    :type l: integer
                    :param parent: parent node
                    :type parent: integer
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
                                phi[input_index][feature_index] += (1 / nb_trees) * w * (node_path[f"node {j}"]["one"][i] - node_path[f"node {j}"]["zero"][i]) * self.trees_value[ind_tree][j]
                            else:
                                for group_index, group in enumerate(self.feature_groups):
                                    if node_path[f"node {j}"]["feature"][i] in group:
                                        break
                                phi[input_index][group_index] += (1 / nb_trees) * w * (node_path[f"node {j}"]["one"][i] - node_path[f"node {j}"]["zero"][i]) * self.trees_value[ind_tree][j]
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
                            iz, io = node_path[f"node {j}"]["zero"][k], node_path[f"node {j}"]["one"][k]
                            unwind(node_path[f"node {j}"], k, l)
                            l -= 1
                        recurse(h, node_path, iz * node_fraction[h]/node_fraction[j], io, node_features[j], l + 1, j)
                        recurse(c, node_path, iz * node_fraction[c]/node_fraction[j], 0, node_features[j], l + 1, j)
                recurse(0, {"node -1": {"weight": np.zeros(tot_length),
                            "zero": np.zeros(tot_length),
                            "one": np.zeros(tot_length),
                            "feature": np.zeros(tot_length)}}, 1, 1, -1, l = 0, parent = -1)
        if phi.shape[0] == 1:
            return phi[0]
        return phi
# Sum shap values of encoded data
def sum_cat_shap(shap_values, feature_groups, n_classes = None):
    """
    Sum shap values of encoded categorical variables when shap values have been computed separately.
    :param shap_values: initial shap values from encoded features
    :type shap_values: np.ndarray
    :param feature_groups: list of feature groups where the shap values needs to be summed up.
    :type feature_groups: list of list of int.
    :param n_classes: if classification: number of classes.
    :type n_classes: int.
    :returns: sum of shap values according to feature_groups.
    :rtype: np.ndarray
    """
    # Get dimension of shap values
    if n_classes == None: # Regression
        if len(shap_values.shape) == 2:
            n_inputs = shap_values.shape[0]
            n_features = shap_values.shape[1]
        else:
            n_inputs = 1
            n_features = len(shap_values)
    else:
        if len(shap_values.shape) == 3:
            n_inputs = shap_values.shape[0]
            n_features = shap_values.shape[1]
        else:
            n_inputs = 1
            n_features = len(shap_values)
    # Check if feature_groups is adapted
    if type(feature_groups) != list:
        raise TypeError("feature_groups must be a list, type(feature_groups): " + str(type(feature_groups)))
    feature_list = []
    # Check types
    groups = []
    for index_group in feature_groups:
        if type(index_group) == int:
            feature_list.append(index_group)
            groups.append([index_group])
        elif type(index_group) == list:
            for ind in index_group:
                if (type(ind) == int) or (type(ind) == np.int32):
                    feature_list.append(ind)
                else:
                    raise TypeError("feature_groups must contain integers or lists of integers:" + str(feature_groups))
            groups.append(index_group)
        else:
            raise TypeError("feature_groups must contain integers or lists of integers" + str(feature_groups))
    # Check number of features
    feature_list = list(dict.fromkeys(feature_list))
    if len(feature_list) != n_features:
        raise Exception("Wrong number of features in feature_groups:")
    if (max(feature_list) != n_features - 1) or (min(feature_list) != 0):
        raise Exception("Wrong feature indices in feature_groups")
    # Aggregate cat shap values
    if n_classes == None: # Regression
        aggregated_shap = np.zeros((n_inputs, len(groups)))
    else: # Classification
        aggregated_shap = np.zeros((n_inputs, len(groups), n_classes))
    for index_group, group in enumerate(groups):
        for feature_index in group:
            aggregated_shap[:,index_group] += shap_values[:, feature_index]
    return aggregated_shap
# Normalise absolute Shap values:
def normalise_absolute_shap_value(shap_values, n_decimals = 3):
    """
    Transform shap values to obtain percentage of absolute value.
    :param shap_values: shap values of samples.
    :type shap_values: np.ndarray with two dimensions (n_samples x n_features).
    :param n_decimals: number of decimals.
    :type n_decimals: int.
    :returns: 1d-array of normalised absolute shap values. 
    :rtype: np.ndarray.
    """
    if len(shap_values.shape) != 2:
        raise Exception("shap_values must be a 2D-array")
    return ((abs(shap_values).T/abs(shap_values).sum(axis = 1)).T).round(n_decimals)

# Bar plot of mean values.
def bar_plot(abs_shap_values, ax, max_features = 10, feature_names = None, n_decimals = 3):
    """
    Create a global feature importance plot by plotting the mean value
    for each feature over all the given samples.
    :param abs_shap_values: (normalised) absolute shap values of samples.
    :type abs_shap_values: np.ndarray with two dimensions (n_samples x n_features).
    :param ax: Axes to plot to.
    :type ax: matplotlib axis.
    :param max_features: maximum features displayed.
    :type max_features: int.
    :param feature_names: feature names.
    :type feature_names: list.
    :param n_decimals: number of decimals.
    :type n_decimals: int.
    :returns: None.
    """
    mean_shap = abs_shap_values.mean(axis = 1).round(n_decimals)
    nb_features = min(max_features, abs_shap_values.shape[1])
    sorted_index = np.argsort(mean_shap)[::-1][:nb_features]
    sorted_mean = mean_shap[sorted_index]
    # Add feature names
    if feature_names == None:
        selected_features = [f"X_{i}" for i in range(nb_features)]
    else:
        selected_features = feature_names[sorted_index]
    # Plot bars
    bars = ax.barh(np.arange(nb_features), sorted_mean, align='center', color = "#b2185d")
    ax.set_yticks(np.arange(nb_features), labels=selected_features)
    ax.bar_label(bars, sorted_mean, padding = 5, color="#b2185d")
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_xlim(right = sorted_mean[0] * 1.1)
