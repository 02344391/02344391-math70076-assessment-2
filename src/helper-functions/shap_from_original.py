import numpy as np
import pandas as pd

class TreeExplainer:
    """A pure Python (slow) implementation of Tree SHAP."""

    def __init__(self, model, feature_groups = None, **kwargs):
        self.model_type = "internal"

        if str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
        elif str(type(model)).endswith("sklearn.ensemble._forest.RandomForestClassifier'>"):
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
        elif str(type(model)).endswith("sklearn.tree._classes.DecisionTreeRegressor'>"): # New
            self.trees = [Tree(model.tree_)] # New
        elif str(type(model)).endswith("sklearn.tree._classes.DecisionTreeClassifier'>"): # New
            self.trees = [Tree(model.tree_)] # New
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            self.model_type = "xgboost"
            self.trees = model
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            self.model_type = "lightgbm"
            self.trees = model
        else:
            raise TypeError("Model type not supported by TreeExplainer: " + str(type(model)))

        if self.model_type == "internal":
            # Preallocate space for the unique path data
            maxd = np.max([t.max_depth for t in self.trees]) + 2
            s = (maxd * (maxd + 1)) // 2
            self.feature_indexes = np.zeros(s, dtype=np.int32)
            self.zero_fractions = np.zeros(s, dtype=np.float64)
            self.one_fractions = np.zeros(s, dtype=np.float64)
            self.pweights = np.zeros(s, dtype=np.float64)
            self.feature_groups = feature_groups # new

    def shap_values(self, X, tree_limit=-1,**kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        # these are about 10x faster than the numba jit'd implementation below...
        if self.model_type == "xgboost":
            import xgboost
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit==-1:
                tree_limit=0
            return self.trees.predict(X, ntree_limit=tree_limit, pred_contribs=True)
        elif self.model_type == "lightgbm":
            return self.trees.predict(X, num_iteration=tree_limit, pred_contrib=True)

        # convert dataframes
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        n_outputs = self.trees[0].values.shape[1]

        # single instance
        if len(X.shape) == 1:
            if self.feature_groups == None: # new
                phi = np.zeros(X.shape[0], n_outputs) # new
            else: # new
                phi = np.zeros(len(self.feature_groups), n_outputs) # new
            for t in self.trees:
                self.tree_shap(t, X, phi)
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, 0]
            else:
                return [phi[:, i] for i in range(n_outputs)]

        elif len(X.shape) == 2:
            if self.feature_groups == None: # new
                phi = np.zeros((X.shape[0], X.shape[1], n_outputs))            # new 
            else: # new
                phi = np.zeros((X.shape[0], len(self.feature_groups), n_outputs)) # new
            for i in range(X.shape[0]):
                for t in self.trees:
                    self.tree_shap(t, X[i,:], phi[i,:,:])
            phi /= len(self.trees)

            if n_outputs == 1:
                return phi[:, :, 0]
            else:
                return [phi[:, :, i] for i in range(n_outputs)]

    def tree_shap(self, tree, x, phi):
        # start the recursive algorithm
        tree_shap_recursive(
            tree.children_left, tree.children_right, tree.children_default, tree.features,
            tree.thresholds, tree.values, tree.node_sample_weight,
            x, phi, 0, 0, self.feature_indexes, self.zero_fractions, self.one_fractions, self.pweights,
            1, 1, -1, self.feature_groups # new
        )


# extend our decision path with a fraction of one and zero extensions

def extend_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, zero_fraction, one_fraction, feature_index):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1.
    else:
        pweights[unique_depth] = 0.

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1.) / (unique_depth + 1.)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1.)

# undo a previous extension of the decision path
#@numba.jit(nopython=True, nogil=True)
def unwind_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1.)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]

# determine what the total permutation weight would be if
# we unwound a previous extension in the decision path

def unwound_path_sum(zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.:
            tmp = next_one_portion * (unique_depth + 1.) / ((i + 1.) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1.))
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1.))
    return total


class Tree:
    def __init__(self, tree, normalize=False):
        if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            if normalize:
                self.values = (tree.value[:,0,:].T / tree.value[:,0,:].sum(1)).T
            else:
                self.values = tree.value[:,0,:]


            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

            # we recompute the expectations to make sure they follow the SHAP logic
            self.max_depth = compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values, 0
            )

def compute_expectations(children_left, children_right, node_sample_weight, values, i, depth=0):
    if children_right[i] == -1:
        values[i,:] = values[i,:]
        return 0
    else:
        li = children_left[i]
        ri = children_right[i]
        depth_left = compute_expectations(children_left, children_right, node_sample_weight, values, li, depth + 1)
        depth_right = compute_expectations(children_left, children_right, node_sample_weight, values, ri, depth + 1)
        left_weight = node_sample_weight[li]
        right_weight = node_sample_weight[ri]
        v = (left_weight * values[li,:] + right_weight * values[ri,:]) / (left_weight + right_weight)
        values[i,:] = v
        return max(depth_left, depth_right) + 1

# recursive computation of SHAP values for a decision tree
def tree_shap_recursive(children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
                        x, phi, node_index, unique_depth, parent_feature_indexes,
                        parent_zero_fractions, parent_one_fractions, parent_pweights, parent_zero_fraction,
                        parent_one_fraction, parent_feature_index, feature_index_groups):
    # stop if we have no weight coming down to us

    # extend the unique path

    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]

    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]
    extend_path(
            feature_indexes, zero_fractions, one_fractions, pweights,
            unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
        )
    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth+1):

            w = unwound_path_sum(zero_fractions, one_fractions, pweights, unique_depth, i)
            if feature_index_groups == None: # new
                phi[feature_indexes[i],:] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index,:]
            else: # new
                for group_index, group in enumerate(feature_index_groups): # new
                    if feature_indexes[i] in group: # new
                        break # new
                phi[group_index,:] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index,:] # new


    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = (cright if hot_index == cleft else cleft)
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1.
        incoming_one_fraction = 1.

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        if feature_index_groups == None: # new
            while (path_index <= unique_depth): 
                if feature_indexes[path_index] == split_index:
                    break
                path_index += 1
        else: # new
            for group_index, group in enumerate(feature_index_groups): # new
                if split_index in group: # new
                    break # new
            while (path_index <= unique_depth): # new
                if feature_indexes[path_index] in feature_index_groups[group_index]: # new
                    break # new
                path_index += 1 # new

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1



        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, feature_index_groups
        )

        tree_shap_recursive(
            children_left, children_right, children_default, features, thresholds, values, node_sample_weight,
            x, phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, feature_index_groups
        )