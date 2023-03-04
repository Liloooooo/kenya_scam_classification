from SCAM_TRAINER import ScamTrainer
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-datadir', type=str, help='directory that contains files train.csv and val.csv, train already augmented if applicable', required=True)
parser.add_argument('-outdir', type=str, default='./model_save/',  help='directory where best trained model is saved. Best refers to model with lowest validation loss')
parser.add_argument('-model', type=str, default = 'electra', choices=['electra', 'bert', 'roberta'])
parser.add_argument('-inter', default=None)
parser.add_argument('-lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('-batch', default=24, type=int, help='batch size')
parser.add_argument('-warmup', default=0.01, type=float, help='warmup ratio')
parser.add_argument('-epochs', default=9, type=int, help='number of epochs')
parser.add_argument('-dropout', default=0.1, type=float, help='dropout ratio in classification layer')
parser.add_argument('-reinit', default=0, type=int, help='number of top layers to re-initialize')
parser.add_argument('-seeds', default=[80, 800, 8000], type=list, help='list of random seeds')

args = parser.parse_args()
arg_dict = {'model_type': args.model,  
        'data': 'standard', 
         'intermediate_task': args.inter, 
         'learning_rate': args.lr, 
         'batch_size': args.batch,
         'warmup_ratio': args.warmup, 
         'num_epochs': args.epochs, 
         'classifier_dropout': args.dropout,
         'reinit_layers': args.reinit}

if __name__ == '__main__':
    input_path = args.datadir
    train = pd.read_csv(input_path + 'train.csv')
    val = pd.read_csv(input_path + 'val.csv')
    trainer = ScamTrainer(arg_dict)
    trainer.fit(train_dataset=train, val_dataset=val, seed=args.seeds)
    trainer.save_best_model(args.outdir)

