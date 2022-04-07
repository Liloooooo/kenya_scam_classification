"""Data preprocessing for scam classification """

import pandas as pd
import numpy as np
import difflib
from sklearn.model_selection import train_test_split
import argparse


def remove_ambiguous(dataframe):
    """Remove ambigous rows. Ambiguous rows are rows in which the orginal
    evaluation differs from the coders' average evaluation.

    Args:
        dataframe: pandas dataframe
    Returns:
        reduced pandas dataframe with one col for target only,
        dataframe containing ambiguous rows

    """

    df = dataframe.copy()
    mask_nan = df.target_check.isna()
    mask_nan_orig = df.target_orig.isna()
    mask_amb = df.target_orig != df.target_check
    df = df.loc[mask_nan_orig | mask_nan | ~mask_amb, :]
    df_amb = df.loc[mask_amb, :].reset_index(drop=True)
    df.target_orig.fillna(df.target_check, inplace=True)
    df = df.loc[:, ["text", "target_orig"]]
    df.reset_index(drop=True, inplace=True)
    print("ambiguous rows:", len(dataframe) - len(df))
    return df, df_amb


def remove_similar(dataframe, cutoff=0.9):
    """Remove similar rows in pandas dataframe. For this purpose, difflib is
    used. Here, similarity is evaluated based on the longest substring
    contained the string to be compared. If the highest similarity to all other
    remaining messages is greater than a given cutoff, this row is removed,
    starting from first row. Never remove similar rows with different labels.
    Print removed messages.

    Args:
        dataframe (pandas dataframe):
            Column with messages to be compared is named 'text'
        cutoff (float, optional):
            ratio between 0 and 1. Defaults to 0.8.
    Returns:
        pandas dataframe with rows removed

    """

    df = dataframe.copy()
    df_iter = dataframe.copy()
    for i in range(len(df_iter)):
        if df.loc[i, "target_orig"] == 0:
            mask = df.loc[:, "target_orig"] == 0
        else:
            mask = df.loc[:, "target_orig"] == 1
        close = (
            len(
                difflib.get_close_matches(
                    df_iter.loc[i, "text"],
                    df.loc[mask, "text"],
                    n=2,
                    cutoff=cutoff,
                )
            )
            - 1
        )
        if close > 0:
            print("deleted: row ", i, ": ", df_iter.loc[i, "text"][:40])
            df.drop(i, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("similar rows deleted: ", len(dataframe) - len(df))
    return df


def add_similarity_score(df_similar, df_compare):
    """For each row in df_similar, add a similarity score that refers to the
    most similar message in df_compare for a given label.
    Similarity is evaluated using the difflib.SequenceMatcher()

    Args:
        df_similar (pandas dataframe): data to be transformed by adding a column,
                                        containing strings in the column 'text'
        df_compare (pandas dataframe): data to be compared, contains strings
                                        in the column 'text'

    Returns:
        pandas dataframe with added column 'highest_similarity'

    """

    df_sim = df_similar.copy()
    mask_sim_0 = df_sim.target_orig == 0
    mask_sim_1 = df_sim.target_orig == 1
    mask_com_0 = df_compare.target_orig == 0
    mask_com_1 = df_compare.target_orig == 1
    df_sim_0 = df_sim.loc[mask_sim_0, :].reset_index(drop=True)
    df_sim_1 = df_sim.loc[mask_sim_1, :].reset_index(drop=True)
    df_com_0 = df_compare.loc[mask_com_0, :].reset_index(drop=True)
    df_com_1 = df_compare.loc[mask_com_1, :].reset_index(drop=True)
    score_list_0, score_list_1 = [], []
    for i in range(len(df_sim_0)):
        best_match = difflib.get_close_matches(
            df_sim_0.loc[i, "text"], df_com_0.loc[:, "text"], n=2, cutoff=0
        )[0]
        score = difflib.SequenceMatcher(
            None, df_sim_0.loc[i, "text"], best_match
        ).ratio()
        score_list_0.append(round(score, 2))
    for i in range(len(df_sim_1)):
        best_match = difflib.get_close_matches(
            df_sim_1.loc[i, "text"], df_com_1.loc[:, "text"], n=2, cutoff=0
        )[0]
        score = difflib.SequenceMatcher(
            None, df_sim_1.loc[i, "text"], best_match
        ).ratio()
        score_list_1.append(round(score, 2))
    df_sim_0["similarity_to_train_set"] = score_list_0
    df_sim_1["similarity_to_train_set"] = score_list_1
    df = pd.concat([df_sim_0, df_sim_1])
    return df.sample(frac=1).reset_index(drop=True)


def process_data(input_path, output_dir="", cutoff=0.9):
    """Data preprocessing function. Read in excel_file containing all data from
    input_path, remove similar and ambiguous rows, split data into training,
    validation and test_set (76.5% / 15% / 8.5%). Add similarity_to_training_set
    column to validation and test set. Save train, val, test data as .csv files
    to output_dir.

    Args:
        input_path (str): path to excel file containing all data with columns
                            'text' and 'target_orig'
        output_dir (str, optional): path to directory where data files will be
                            saved. Defaults to empty string.

    """

    np.random.seed(100)
    df = pd.read_excel(input_path, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df = remove_similar(df, cutoff=cutoff)
    df = remove_ambiguous(df)[0]
    df_train, df_valid = train_test_split(df, test_size=0.15)
    df_train.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(df_train, test_size=0.1)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_valid = add_similarity_score(df_valid, df_train)
    df_test = add_similarity_score(df_test, df_train)
    df_train.to_csv(output_dir + "train.csv", index=False)
    df_valid.to_csv(output_dir + "val.csv", index=False)
    df_test.to_csv(output_dir + "test.csv", index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=False)
parser.add_argument("--input_path", type=str, required=True)
args = parser.parse_args()
if args.output_dir:
    process_data(input_path=args.input_path, output_dir=args.output_dir)
else:
    process_data(input_path=args.input_path)


###USAGE in command line
# python data_preprocessing.py --input_path PATH output_dir DIR
