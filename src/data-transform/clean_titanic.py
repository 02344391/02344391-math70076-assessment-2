# Import modules
import os
import pandas as pd

# Import data
input_path = os.path.abspath("../../data/raw/titanic.csv")
titanic = pd.read_csv(input_path)[["survived", 
                                    "pclass", 
                                    "adult_male", 
                                    "age", 
                                    "sibsp", 
                                    "parch",
                                    "fare", 
                                    "embarked"]]
# Remove rows with NaN values
titanic = titanic.dropna().reset_index(drop = True)

# Encode pclass and embarked
titanic = pd.get_dummies(titanic, 
                         columns= ["pclass", "embarked"], 
                         drop_first= True)
# Save data
output_path = os.path.abspath("../../data/derived/titanic_cleaned.csv")
titanic.to_csv(output_path, index = False)

