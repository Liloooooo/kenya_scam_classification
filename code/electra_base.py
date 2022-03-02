#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:32:52 2022

@author: lilowagner
"""


#import tensorflow as tf 
import torch
import pandas as pd 
import numpy as np
import random
import time
from transformers import AutoTokenizer, ElectraForSequenceClassification, get_linear_schedule_with_warmup, ElectraTokenizer, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import os
import json


def preprocessing_2(dataset):
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator", 
                                              do_lower_case=True)
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
    print(labels.dtype)
    dataset_train = TensorDataset(input_ids, attention_masks, labels)
    return dataset_train

    
def dataloaders(train_data, val_data, args): 
    #train_data, val_data in TensorDataset format (use preprocessing function first)
    random_state = args['seed']
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    batch_size = args['batch_size']
    train_dataloader = DataLoader(
                train_data,  # The training samples.
                sampler = RandomSampler(train_data), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    validation_dataloader = DataLoader(
                val_data, # The validation samples.
                sampler = SequentialSampler(val_data), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader, validation_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def training(train_dataloader, val_dataloader, args, device): 
    random_state = args['seed']
    warmup_ratio = args['warmup_ratio']
    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-base-discriminator", classifier_dropout = args['classifier_dropout']
    )

    model.cuda()
    optimizer = AdamW(model.parameters(),
                      lr = args['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    epochs = args['num_epochs']
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = round(warmup_ratio*total_steps,0), 
                                                num_training_steps = total_steps)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader): #len(train_dataloader) = len(train_dataset)//16
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Always clear any previously calculated gradients before performing a backward pass. 
            model.zero_grad()        
            # Perform a forward pass (evaluate the model on this training batch).
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           labels=b_labels,
                           return_dict=True)
            loss = result.loss #tensor containing single value (get with .item() method)
            logits = result.logits
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() 
            scheduler.step()# Update the learning rate.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = time.time() - t0 #training time for per epoch
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training for epoch {} took: {:}".format(epoch_i, training_time))
        # Validation metrics
        print("")
        print('Validation...')
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        total_val_accuracy = 0
        total_val_loss = 0
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        #no gradient computation needed when evaluating model
                result = model(b_input_ids, 
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
            total_val_accuracy += flat_accuracy(logits, label_ids)
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
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time
            }
        )
    print("")
    print("Training completed")
    print("Total training took {:} (h:mm:ss)".format(time.time()-total_t0))
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
    
    
# args = {'model_type': 'bert_base', 
#         'learing_rate': 2e-5, 
#         'batch_size': 16, 
#         'weight_decay': 0, 
#         'seed': 0, 
#         'data_dir': '../data/', 
#         'warmup_ratio': 0.1, 
#         'num_epochs': 4, 
#         'augment_data: 0, 
#         'classifier_dropout': None} #None equals 0.1 

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



