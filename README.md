# Scam classification of text messages from Kenya

Text based scam classification using one of the following pretrained models: roberta, bert, electra. 
The original data is not provided here. 
 



- code/data_preprocessing: 
     - remove ambiguous and very similar data, splits data into train, validation and test set, adds similarity column to train and test set (for         
       interpretation purposes), and saves data as .csv file to some output directory. Input data must be in .xlsx format and have two columns, 'text'  
       and 'target_orig'. 
- code/augment_data: data augmentation, using either contextual or vocabulary based synonym replacement 
- code/scam_trainer.py: trains model according to defined model configurations, saves model and training statistics
- code/scam_evaluator.py: evaluates test set and predicts labels and probabilities 



