# Scam classification of text messages from Kenya

Text based scam classification using one of the following pretrained models: roberta, bert, electra. 
The original data is not provided here. 
The training code was tuned following recommendations for few-sample BERT models:
<a href="https://arxiv.org/abs/2006.05987" target="_blank">Zhang et. al: Revisiting Few-sample BERT Fine-tuning</a>.
 



- code/data_preprocessing.py: 
     - remove ambiguous and very similar data, split data into train, validation and test set, add similarity column to train and test set and saves  data as .csv files to some output directory. 
     - input data must be in .xlsx format and have two columns, 'text'  
       and 'target_orig'. 
- code/augment_data: data augmentation, using either contextual or vocabulary based synonym replacement 
- code/scam_trainer.py: trains model according to defined model configurations, saves model and training statistics
- code/scam_evaluator.py: evaluates test set, predicts labels and probabilities and calculates Shapley values 

## Usage

### Data preprocessing 

In command line, where INPUT_PATH is the path to an .xlsx file with columns 'text' and 'target_orig' (Labels in 0/1 format):  

`python code/data_preprocessing.py --input_path INPUT_PATH --output_dir OUTPUT_DIRECTORY`

### Training 

The code can be run from the terminal using
python code/task.py -datadir INPUT_DIR [options]`
where INPUT_DIR is the path to a directory containing the files 'train.csv' and 'val.csv'.
Other optional options are: 

```
-outdir <string>       path to directory where output files will be saved 
                       [default: './model_save/']

-model <string>        model architecture, choices are 'electra', 'roberta', 'bert'
                       [default: 'electra']

-lr <float>            learning rate
                       [default: 2e-5]

-batch <int>           batch size
                       [default: 24]

-warmup <float>        warmup ratio 
                       [default: 0.01]
                       
-epochs <int>          number of training epochs 
                       [default: 9]
                                         
-dropout <float>       dropout ratio in classification layer
                       [default: 0.1]  

-reinit <int>          number of top layers to re-initialize
                       [default: 0] 
                       
-seeds <list>          list of random seeds, and from which best option is chosen when saving model
                       [default: [80, 800, 8000]]
```

Alternatively, SCAM_TRAINER can be fitted using a dictionnary, for instance: 

```
from SCAM_TRAINER import ScamTrainer
import pandas as pd

train = pd.read_csv(PATH_TO_TRAIN_FILE)
val = pd.read_csv(PATH_TO_VALIDATION_FILE)

args = {'model_type': 'electra',  
        'data': 'standard', 
         'intermediate_task': None, 
         'learning_rate': 2e-5, 
         'batch_size': 24,
         'warmup_ratio': 0.01, 
         'num_epochs': 9, 
         'classifier_dropout': 0.1,
         'reinit_layers': 0}

trainer = ScamTrainer(args)
trainer.fit(train_dataset=train, val_dataset=val, seed=[80, 800, 8000])
trainer.save_best_model(PATH_TO_SAVE_DIR)
```

### Evaluation/Prediction

```
from scam_evaluator import ScamEvaluator
import pandas as pd

evaluator = ScamEvaluator(model_dir)
test = pd.read_csv('test.csv')
_, test_acc = evaluator.evaluate(test, feature_col='text', target_col='target_orig')
predictions = evaluator.predict_proba(test, feature_col='text', target_col='target_orig')
```

Alternatively, ScamEvaluator.predict_proba and ScamEvaluator.predict_labels takes as input a list: 
```
phrases = ['Dear member your account has been (suspended). To unlock, call/sms 0799 096453', 
           'Dear member your account has been (suspended). To unlock, go to our app and unlock']
           
evaluator.predict_proba(phrases)
```

To calculate coalitional Shapley values: 
```
shap_values = evaluator.shap_values(test.loc[:, 'text'])
```
or 
```
shap_values = evaluator.shap_values(phrases)
```

## Shapley values

Shapley values are calculated using the shap PartitionExplainer. 
Tokenization is done using transformers.BasicTokenizer, which is the tokenizer used in Bert/Electra, prior to WordPiece tokenization. 

The base value is set to the predicted probability of the empty string. 
