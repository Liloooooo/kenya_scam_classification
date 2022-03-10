"""
Main module for training and evaluating scams. For training, GPU should be available. 
"""

from scam_evaluator import ScamTrainer
from scam_evaluator import ScamEvaluator
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_dir = PATH  # insert path to data files
train = pd.read_csv(data_dir + "train.csv")
val = pd.read_csv(data_dir + "val.csv")
test = pd.read_csv(data_dir + "test.csv")
args = {
    "model_type": "electra",  # choose from ['bert', 'electra', 'roberta']
    "intermediate_task": None,  # None/0 or 'yes'/1
    "learning_rate": 2e-5,
    "batch_size": 32,
    "warmup_ratio": 0.1,
    "num_epochs": 8,
    "classifier_dropout": 0.1,
    "reinit_layers": 0,
}

trainer = ScamTrainer(args)
trainer.fit(train_dataset=train, val_dataset=val, seed=[50])
trainer.save_best_model()

###to evaluate
# evaluator = ScamEvaluator('./model_save/')
# test_loss, test_acc = evaluator.evaluate(test)
# predicted_labels = evaluator.predict_labels(test)
# predicted_proba = evaluator.predict_proba(test)
# cm = confusion_matrix(test.target_orig, predicted_labels)
# ConfusionMatrixDisplay(cm).plot()
