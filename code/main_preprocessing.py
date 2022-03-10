"""
Main module for preprocessing data. 
"""

from preprocessing import (
    remove_ambiguous,
    remove_similar,
    add_similarity_score,
)
from augment_data import augment_text
import pandas as pd
from sklearn.model_selection import train_test_split

path = DATA_PATH  # replace
file = "data.xlsx"  # replace by file
df = pd.read_excel(path + "/" + file, index_col=0).reset_index(drop=True)
df = remove_similar(df)  # remove similar, but with high cutoff 0.8
df = remove_ambiguous(df)
df_train, df_valid = train_test_split(df, test_size=0.2)

df_train.reset_index(drop=True, inplace=True)
# df_test.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)

# df_test = add_similarity_score(df_test, df_train)  # for demonstration purposes only
df_valid = add_similarity_score(
    df_valid, df_train
)  # for demonstration purposes --> better ability to evaluate metrics
# df_valid = remove_similar(df_valid) # very similar texts are not evaluated multiple times

print(df_train.shape, df_valid.shape)
