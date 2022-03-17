# Scam classification of text messages from Kenya

- code/data_preprocessing: remove ambiguous and very similar data, splits data into train, validation and test set, adds similarity column to train and test set (for interpretation purposes), and saves data as .csv file to some output directory 
- code/augment_data: data augmentation, using either contextual or vocabulary based synonym replacement 
- code/scam_trainer.py: trains model according to defined model configurations, saves model and training statistics
- code/scam_evaluator.py: evaluates test set and predicts labels and probabilities 



