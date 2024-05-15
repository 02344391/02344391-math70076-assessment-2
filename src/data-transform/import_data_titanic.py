# Import titanic dataset from seaborn
"""
(requires internet)
"""
# Import modules
import os
import seaborn as sns

# import and save raw data

output_path = os.path.abspath("../../data/raw/titanic.csv")
df = sns.load_dataset('titanic')
df.to_csv(output_path)