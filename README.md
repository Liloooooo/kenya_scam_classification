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

`python data_preprocessing.py --input_path INPUT_PATH --output_dir OUTPUT_DIRECTORY`

