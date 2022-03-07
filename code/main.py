#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:46:54 2022

@author: lilowagner
"""

from SCAM_TRAINER import ScamTrainer
import pandas as pd 

data_dir = PATH #insert path to data files
train = pd.read_csv(data_dir + 'train.csv')
val = pd.read_csv(data_dir + 'val.csv')
args = {'model_type': 'electra', #choose from ['bert', 'electra', 'roberta']
         'intermediate_task': None, #None/0 or 'yes'/1
         'learning_rate': 2e-5, 
         'batch_size': 32, 
         'warmup_ratio': 0.1, 
         'num_epochs': 8, 
         'classifier_dropout': 0.1,
         'reinit_layers': 0}

trainer = ScamTrainer(args)
trainer.fit(train_dataset=train, val_dataset=val, seed=[50])
trainer.save_best_model()
