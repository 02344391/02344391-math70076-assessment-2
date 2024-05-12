# Script to plot decision tree with chosen colours

import matplotlib.pyplot as plt
import pandas as pd
def plot_tree_color(ax, tree, color = None):
    """
    """
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    samples = tree.tree_.n_node_samples
    values = tree.tree_.value.flatten()
    lefts = tree.tree_.children_left
    rights = tree.tree_.children_right
    max_depth = tree.tree_.max_depth
    tree_dict = {}
    def plot_recurse_tree(j, depth, leaf):
        """
        """
        if depth != max_depth + 1:
            if leaf:
                text = None
                if f"depth {depth}" in tree_dict:
                    tree_dict[f"depth {depth}"]["text"].append(text)
                    if color == None:
                        tree_dict[f"depth {depth}"]["color"].append(color)
                    else:
                        tree_dict[f"depth {depth}"]["color"].append(color[j])        
                else : 
                    tree_dict[f"depth {depth}"] = {}
                    tree_dict[f"depth {depth}"]["text"] = [text]
                    if color == None:
                        tree_dict[f"depth {depth}"]["color"] = [color]
                    else:
                        tree_dict[f"depth {depth}"]["color"] = color[j]
            else:
                leaf = lefts[j] < 0
                text = str(features[j]) + r"$\leq$" + str(thresholds[j])
                text += "\n" + "n =  " + str(values[j])
                if leaf:
                    text += "\n" + "value =  " + str(samples[j])
                if f"depth {depth}" in tree_dict:
                    tree_dict[f"depth {depth}"]["text"].append(text)
                    if color == None:
                        tree_dict[f"depth {depth}"]["color"].append(color)
                    else:
                        tree_dict[f"depth {depth}"]["color"].append(color[j])        
                else : 
                    tree_dict[f"depth {depth}"] = {}
                    tree_dict[f"depth {depth}"]["text"] = [text]
                    if color == None:
                        tree_dict[f"depth {depth}"]["color"] = [color]
                    else:
                        tree_dict[f"depth {depth}"]["color"].append(color[j])
            plot_recurse_tree(lefts[j], depth + 1, leaf)
            plot_recurse_tree(rights[j], depth + 1, leaf)
    plot_recurse_tree(0, 0, False)
    for key in tree_dict:
        print(tree_dict[key])




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

plot_tree_color(1, Tree, color = None)


