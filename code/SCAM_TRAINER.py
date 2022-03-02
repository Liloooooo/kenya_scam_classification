#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:32:52 2022

@author: lilowagner
"""


#import tensorflow as tf 
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import random
import time
from transformers import BertForSequenceClassification, ElectraForSequenceClassification, RobertaForSequenceClassification, get_linear_schedule_with_warmup, BertTokenizer, ElectraTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import os
import json




class ScamTrainer:
    def __init__(self, args): 
        self.model_type = args['model_type'].lower()
        self.int_task = args['intermediate_task']
        self.learning_rate = args['learning_rate']
        self.batch_size = args['batch_size']
        self.seed = args['seed']
        self.warmup_ratio = args['warmup_ratio']
        self.num_epochs = args['num_epochs']
        self.classifier_dropout = args['classifier_dropout']
        self.reinit_layers = args['reinit_layers']
        assert self.model_type in ['bert', 'electra', 'roberta']
        assert isinstance(self.learning_rate, float)
        if not 0 < self.learning_rate < 1: 
            raise ValueError
        assert isinstance(self.batch_size, int)
        if self.batch_size < 0: 
            raise ValueError
        assert isinstance(self.warmup_ratio, float)
        if not 0 <= self.warmup_ratio < 1: 
            raise ValueError
        assert isinstance(self.num_epochs, int)
        assert isinstance(self.classifier_dropout, float)
        if not 0 <= self.classifier_dropout < 1: 
            raise ValueError
        if self.reinit_layers: 
            assert isinstance(self.reinit_layers, int)
            if self.reinit_layers < 0: 
                raise ValueError
        
        self.model = None
        self.device = torch.device['cuda'] if torch.cuda.is_available() else torch.device('cpu')
        if self.device == torch.device['cuda']:
            print('The following GPU is used:', torch.cuda.get_device_name(0))
        else: 
            print('No GPU available, CPU is used.')

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
    def preprocessing(self, dataset):
        if self.model_type == 'bert': 
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.model_type == 'electra': 
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        else: 
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        encoded_data = tokenizer.batch_encode_plus(
            dataset.text.values, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=128, 
            return_tensors='pt'
        )
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(dataset.target_orig.values).long()
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
        return tensor_dataset

    
    def dataloaders(self, train_data, val_data): 
        #train_data, val_data in TensorDataset format (use preprocessing function first)
        self.set_seed()
        train_dataloader = DataLoader(
                    train_data,  # The training samples.
                    sampler = RandomSampler(train_data), # Select batches randomly
                    batch_size = self.batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    val_data, # The validation samples.
                    sampler = SequentialSampler(val_data), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                )
        return train_dataloader, validation_dataloader

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat) 
    
    def init_model(self): 
        if self.model_type == 'bert': 
            if not self.int_task: 
                return BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", classifier_dropout = self.classifier_dropout)
            else: 
                return BertForSequenceClassification.from_pretrained(
                    "ishan/bert-base-uncased-mnli", classifier_dropout = self.classifier_dropout)
        if self.model_type == 'electra': 
            if not self.int_task: 
                return ElectraForSequenceClassification.from_pretrained(
                    'google/electra-base-discriminator', classifier_dropout = self.classifier_dropout)
            else: 
                return ElectraForSequenceClassification.from_pretrained(
                    "howey/electra-base-mnli", classifier_dropout = self.classifier_dropout)
        if self.model_type == 'roberta': 
            if not self.int_task: 
                return RobertaForSequenceClassification.from_pretrained('roberta-base', classifier_dropout = self.classifier_dropout)
            else: 
                return RobertaForSequenceClassification.from_pretrained('textattack/roberta-base-MNLI', classifier_dropout = self.classifier_dropout)

    def _reinit_layer_weights(self): 
        if not self.model: 
            raise NotImplementedError
        encoder_temp = getattr(self.model, self.model_type) #encoder_temp = BertModel
        if self.model_type in ["bert", "roberta"]: #apparently, electra does not have a pooler 
            print('reinitializing pooler...')
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        for layer in encoder_temp.encoder.layer[-self.reinit_layers :]:
            print('reinitializing layer {}...'.format(layer))
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
                    
    def training(self, train_data, val_data): 
        self.set_seed()
        self.model = self.init_model()
        if self.reinit_layers: 
            if self.reinit_layers > 0:
                self._reinit_layer_weights()
        self.model.cuda()
        optimizer = AdamW(self.model.parameters(),
                          lr = self.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        epochs = self.num_epochs
        train_dataloader, val_dataloader = self.dataloaders(train_data, val_data)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = round(self.warmup_ratio*total_steps,0), 
                                                    num_training_steps = total_steps)
        training_stats = []
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            total_train_loss = 0
            self.model.train() 
            for step, batch in enumerate(train_dataloader): #len(train_dataloader) = len(train_dataset)//16
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                self.model.zero_grad()        
                result = self.model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask, 
                               labels=b_labels,
                               return_dict=True)
                loss = result.loss #tensor containing single value (get with .item() method)
                logits = result.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step() 
                scheduler.step()# Update the learning rate.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("")
            print('Validation...')
            self.model.eval()
            total_val_accuracy = 0
            total_val_loss = 0
            for batch in val_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                with torch.no_grad():        #no gradient computation needed when evaluating model
                    result = self.model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict=True)
                loss = result.loss
                logits = result.logits
                total_val_loss += loss.item()
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_val_accuracy += self.flat_accuracy(logits, label_ids)
            avg_val_accuracy = total_val_accuracy / len(val_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            avg_val_loss = total_val_loss / len(val_dataloader)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy
                }
            )
        print("")
        print("Training completed")
        training_summary = pd.DataFrame(training_stats).set_index('epoch')
        output_dir = './model_summary/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model_summary to %s" % output_dir)
        #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save.save_pretrained(output_dir)
        training_summary.to_excel(os.path.join(output_dir, 'training_summary.xlsx'))
        #AutoTokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        #torch.save(training_summary, os.path.join(output_dir, 'training_summary.bin'))
        with open(os.path.join(output_dir, 'training_arguments.json'), 'w') as file:
            json.dump(args, file)
            
    
    def save_model(model): 
        output_dir = './model_save/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        

    
# if torch.cuda.is_available():      
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

# print('current working directory:', os.getcwd())
    
    
# args = {'model_type': 'bert', #choose from ['bert', 'electra', 'roberta']
#         'intermediate_task': None, #alternatively, give some value like 1, or 'yes' 
#         'learing_rate': 2e-5, 
#         'batch_size': 16, 
#         'weight_decay': 0, 
#         'seed': 0, 
#         'data_dir': '../data/', 
#         'warmup_ratio': 0.1, 
#         'num_epochs': 4, 
#         'augment_data: 0, 
#         'classifier_dropout': None
#         'reinit_layers': 0}

# #load data 
# data_dir = args['data_dir']
# train_data = 'train.csv'
# val_data = 'val.csv'
# train = pd.read_csv(data_dir + train_data)
# val = pd.read_csv(data_dir + val_data)

# train = preprocessing(train)
# val = preprocessing(val)

# train_dataloader, val_dataloader = dataloaders(train, val, batch_size=args['batch_size'])

# training(train_dataloader, val_dataloader, args, device)



