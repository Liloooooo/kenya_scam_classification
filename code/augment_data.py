#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:49:31 2021

@author: lilowagner
"""

### script to augment cleaned data and save train, validation and test files
### run on colab!

import pandas as pd
import numpy as np
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


def augment_text(
    dataset,
    augmentation_type="contextual",
    samples_0=0,
    samples_1=0,
    aug_p=0.2,
    seed=10,
):
    """Text augmentation by synonym replacement.

    Args
    ----------
    dataset : pandas.DataFrame
        Columns are named 'text' and 'target_orig' (labels: 0/1)
    augmentation_type : str, optional
        'contextual': synonym replacement based on contextual word representations in bert-base-uncased,
        'synonym_dict': substituted words are taken from WordNet.
        Defaults to 'contextual'.
    samples_0 : int, optional
        Number of 0-labelled texts to add to original dataset. Defaults to 0.
    samples_1 : int, optional
        Number of 1-labelled texts to add to original dataset. Defaults to 0.
    aug_p : float, optional
        Share of words to replace. Defaults to 0.2.
    seed : int, optional
        random seed. Defaults to 10.

    Returns
    -------
    pandas.DataFrame: augmented dataset, including original dataset

    """

    assert isinstance(dataset, pd.DataFrame)
    assert isinstance(augmentation_type, str)
    if not augmentation_type in ["contextual", "synonym_dict"]:
        raise ValueError(
            'augmentation_type must be "contextual" or "synonym_dict"'
        )
    assert isinstance(samples_0, int)
    assert isinstance(samples_1, int)
    assert isinstance(aug_p, float)
    if not 0 <= aug_p <= 1:
        raise ValueError
    assert isinstance(seed, int)
    np.random.seed(seed)
    df = dataset.copy()
    df_0 = df[df.target_orig == 0].sample(frac=1).reset_index(drop=True)
    df_1 = df[df.target_orig == 1].sample(frac=1).reset_index(drop=True)
    new_text_0, new_text_1 = [], []
    if augmentation_type == "contextual":
        aug = naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased", action="substitute", aug_p=aug_p
        )
    else:
        aug = naw.SynonymAug(aug_src="wordnet", aug_p=aug_p)
    for i in tqdm(np.random.randint(0, len(df_0), samples_0)):
        text = df_0.loc[i, "text"]
        augmented_text = aug.augment(text)
        new_text_0.append(augmented_text)
    for i in tqdm(np.random.randint(0, len(df_1), samples_1)):
        text = df_1.loc[i, "text"]
        augmented_text = aug.augment(text)
        new_text_1.append(augmented_text)
    new_0 = pd.DataFrame({"text": new_text_0, "target_orig": 0})
    new_1 = pd.DataFrame({"text": new_text_1, "target_orig": 1})
    df = df.append(new_0).append(new_1).sample(frac=1).reset_index(drop=True)
    return df
