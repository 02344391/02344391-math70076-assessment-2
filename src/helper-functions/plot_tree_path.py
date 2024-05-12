# Script to plot decision tree with chosen colours

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Colour Palette used
color_list = sns.color_palette("GnBu_r")

def plot_tree_color(ax, tree, x_input, feature_names, known_features):
    """
    Colour the path of an input descending a tree given known features.
    :param ax: Axes to plot to.
    :type ax: matplotlib axis.
    :param tree: tree model.
    :type tree: sklearn.tree.DecisionTreeRegressor;
    sklearn.tree.DecisionTreeClassifier.
    :param x_input: input to test.
    :type x_input: list or np.ndarray.
    :param feature_names: all the features of the model in order.
    :type feature_names: list of str.
    :param known_features: Subset of known features.
    :type known_features: list or np.ndarray.
    :returns: None 
    """
    plot = plot_tree(tree, 
                     ax=ax, 
                     impurity = False,
                     feature_names = feature_names)
    for node in plot:
        node.set_bbox({"boxstyle": 'round'})
        node.set_backgroundcolor("#C0C0C0")
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    def colour_node(node_index, colour_index):
        """
        Colour node the algorithm passes through.
        :param node_index: Index of the node the algorithm passes through.
        :type node_index: int.
        :param colour_index: Index of the colour the node cell is coloured.
        :type coulour_index: int.
        """
        plot[node_index].set_backgroundcolor(color_list[colour_index])
        plot[node_index].set_color("w")
        # check if internal
        if lefts[node_index] != -1:
            if int(features[node_index]) in known_features:
                if x_input[int(features[node_index])] <= thresholds[node_index]:
                    colour_node(lefts[node_index], colour_index)
                else:
                    colour_node(rights[node_index], colour_index)
            else:
                colour_index +=1
                colour_node(rights[node_index], min(colour_index, len(color_list) + 1))
                colour_node(lefts[node_index], min(colour_index, len(color_list) + 1))
    colour_node(0, 0)

from sklearn.tree import DecisionTreeRegressor
Tree = DecisionTreeRegressor()

target = [-10] * 10 + [10] * 5 + [5] * 20 + [14] * 5
cat_1 = [1] * 10 + [0] * 30
x = [60] * 5 + [45] * 30 +  [50] * 5
cat_2 = [1] * 15 + [0] * 25
data = pd.DataFrame({"cat_1": cat_1,
                     "cat_2": cat_2,
                     "x": x,
                     "target": target})
Tree.fit(data.values[:,:-1], data.values[:,-1])
input_shap = [0,1,60]
fig, ax = plt.subplots(1)
plot_tree_color(ax, Tree, input_shap, data.columns[:-1], [0,1])
plt.savefig(os.path.abspath("../../outputs/plot_tree/figures/tree1.pdf"))

