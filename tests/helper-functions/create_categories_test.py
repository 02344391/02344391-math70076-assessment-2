# Test create_categories

## Import libraries and functions
import sys
import os
import numpy as np

## Import script 
sys.path.append(os.path.abspath("../../src/helper-functions"))
import create_categories

# Generate samples with normal numeric features
seed = 2344391
np.random.seed(seed)
X = np.random.randn(10)

X_cat = create_categories.transform_num_to_cat(X = X, 
                                               feature_name = "X", 
                                               nb_cat = 3, 
                                               cat_names = ["red", "green", "blue"], 
                                               randomness= 0.5, 
                                               seed = seed)
# Test
np.testing.assert_array_equal(X_cat,
                              np.array(["red",
                                        "blue",
                                        "green",
                                        "green",
                                        "red",
                                        "blue",
                                        "green",
                                        "blue",
                                        "red",
                                        "blue"]))