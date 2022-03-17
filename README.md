# Scam classification of text messages from Kenya

Text based scam classification using one of the following pretrained models: roberta, bert, electra. 
The original data is not provided here. 
 



- code/data_preprocessing.py: 
     - remove ambiguous and very similar data, split data into train, validation and test set, add similarity column to train and test set and saves  data as .csv files to some output directory. 
     - input data must be in .xlsx format and have two columns, 'text'  
       and 'target_orig'. 
- code/augment_data: data augmentation, using either contextual or vocabulary based synonym replacement 
- code/scam_trainer.py: trains model according to defined model configurations, saves model and training statistics
- code/scam_evaluator.py: evaluates test set and predicts labels and probabilities 

## Usage

### Data preprocessing 

In command line: 
`python data_preprocessing.py --input_path INPUT_PATH --output_dir OUTPUT_DIRECTORY`

### Training 

```
from SCAM_TRAINER import ScamTrainer
import pandas as pd

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

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
```

### Evaluation 

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



