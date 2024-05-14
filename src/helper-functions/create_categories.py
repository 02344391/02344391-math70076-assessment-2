# Script to generate (randomly or not) categorical features based on numeric features
import numpy as np
def transform_num_to_cat(X, feature_name = None, nb_cat = 2, cat_names = None,  randomness = 0, seed = None):
    """
    Create array of categories from an array of numeric data.
    Categories correspond to a breakdown of numerical data arranged in order.
    Randomness controls the error rate.
    :param X: numerical array to transform.
    :type X: np.ndarray.
    :param feature_name: feature name.
    :type feature_name: list of str.
    :param nb_cat: number of categories.
    :type nb_cat: int.
    :param cat_names: category names.
    :type cat_names: list of str.
    :param randomness: error rate: 0 for a perfect category allocation 
    and 1 for total random allocation.
    :type randomness: float between 0 and 1.
    :param seed: Numpy random state.
    :type seed: int.
    :returns: array of categories.
    :rtype: np.ndarray.
    """
    def choose_cat(true_cat, prob, cat_names):
        """
        Keep true_cat with proba prob and choose 
        another category in cat_names with proba 1-prob.
        :param true_cat: initial category.
        :type true_cat: str.
        :param prob: probability of keeping the right category.
        :type prob: float.
        :param cat_names: list of category names.
        :type cat_names: list of str.
        :returns: new category.
        :rtype: str.
        """
        categories = cat_names.copy()
        if np.random.rand() > prob:
            categories.remove(true_cat)
            return np.random.choice(categories)
        return true_cat
    # Check dimension
    if cat_names != None:
        if len(cat_names) != nb_cat:
            raise Exception("nb_cat must be the number of categories in cat_names")
    else:
        # Create category names
        if feature_name == None:
            cat_names = [f"cat_{i}" for i in range(1, nb_cat+1)]
        else:
            if type(feature_name) != str:
                raise TypeError("feature_name must be str: " + str(type(feature_name)))
            cat_names = [feature_name + f"_cat_{i}" for i in range(1, nb_cat+1)]
    # Normalise X with min max
    min_max_X = (X - min(X))/(max(X) - min(X))
    # Assign (non randomnly) a category according to the value of X
    condlist = [(i / nb_cat <= min_max_X) & (min_max_X <= (i+1)/nb_cat) for i in range(nb_cat)]
    choicelist = cat_names
    X_cat =  np.select(condlist, choicelist)
    if randomness == 0:
        return X_cat
    else:
        # Add randomness
        if (randomness <= 1) and (randomness > 0):    
            if seed != None:
                np.random.seed(seed)
            proba_cat = (1 + (1 - randomness) * (nb_cat - 1)) / nb_cat
            return np.vectorize(lambda a: choose_cat(a, proba_cat, cat_names))(X_cat)
        else:
            raise Exception("randomness must be a float in [0,1]")
