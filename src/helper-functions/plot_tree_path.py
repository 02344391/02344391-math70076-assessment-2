# Script to plot decision tree with chosen colours

from sklearn.tree import plot_tree
import seaborn as sns
import numpy as np

# Colour Palette used
color_list = sns.color_palette("GnBu_r")

def plot_tree_color(ax, tree, x_input, feature_names, known_features, title = True):
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
    :param title: Whether to display input params and known features in title.
    :type title: boolean.
    :returns: None 
    """
    plot = plot_tree(tree, 
                     ax=ax, 
                     impurity = False,
                     feature_names = feature_names)
    if title:
        ax.set_title(f"Input: " + str(x_input) +f";\n known feature(s): {np.array(feature_names)[known_features]}")
    for node in plot:
        node.set_bbox({"boxstyle": 'round'})
        node.set_backgroundcolor("#C0C0C0")
        node.set_text(node.get_text().replace("samples", "n"))
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
            # remove value in text if internal
            plot[node_index].set_text(plot[node_index].get_text().split("value")[0][:-1])
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
    # rearrange the text to make it more coherent
    for node in plot:
        if "value" in node.get_text():
            list_text = node.get_text().split("\n")
            node.set_text(list_text[1] + "\n" + list_text[0])