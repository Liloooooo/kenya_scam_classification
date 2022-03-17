"""Module scam_trainer -- class ScamTrainer: train, document and save Bert-like pytorch model."""


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    ElectraTokenizer,
    RobertaTokenizer,
)
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.optim import AdamW
import os
import json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScamTrainer:
    """Classification for scam/non-scam SMS.

    Attrs:
        args (dict): contains specifications of training parameters, namely:
                - model_type (str, optional):
                        ['bert', 'electra', 'roberta'].
                        Defaults to 'bert'.
                - intermediate_task (str or integer, optional):
                        If 'yes', initialized weights correspond to
                        intermediate task training on MNLI task.
                        Defaults to None.
                - learning_rate (float, optional):
                        Defaults to 2e-5.
                - batch_size (int, optional):
                        Defaults to 32.
                - warmup_ratio (float, optional):
                        Defaults to 0.1.
                - num_epochs (int, optional):
                        Number of epochs that model is trained.
                        Defaults to 4.
                - classifer_dropout (float, optional):
                        droupout ratio in classification layer.
                        Defaults to 0.1
                - reinit_layers (int, optional):
                    number of top layers to reinitialize.
                    If greater than 0, pooling layer is also initialized.
                    Defaults to 0.

    Methods:
        fit(train_dataset, val_dataset, feature_col='text', target_col='target_orig', seed=100):
            train model on train_dataset, and records accuracy and loss of 
            both train_dataset and val_dataset. Can be trained on different seeds.
        save_best_model(output_dir = './model_save/'):
            save best model among trained models.

    """

    def __init__(self, args={}):
        assert isinstance(args, dict), "args must be of type dict"
        self.args = args
        self.model_type = args.get("model_type", "bert").lower()
        self.int_task = args.get("intermediate_task", None)
        self.learning_rate = args.get("learning_rate", 2e-5)
        self.batch_size = args.get("batch_size", 32)
        self.warmup_ratio = args.get("warmup_ratio", 0.1)
        self.num_epochs = args.get("num_epochs", 4)
        self.classifier_dropout = args.get("classifier_dropout", 0.1)
        self.reinit_layers = args.get("reinit_layers", 0)
        assert self.model_type in ["bert", "electra", "roberta"]
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
        self.tokenizer = None
        self.best_model = None
        self.best_loss = float("inf")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("The following GPU is used:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("No GPU available, CPU is used.")

    def _preprocessing(
        self, dataset, feature_col="text", target_col="target_orig"
    ):
        # tokenizes text in pandas dataframe and returns a tensor dataset with columns input_ids, attention_mask and labels (if with_labels)
        assert isinstance(
            dataset, pd.DataFrame
        ), "dataset must be a pandas DataFrame."
        assert (
            feature_col in dataset.columns
        ), "feature_col not found in dataset"
        assert target_col in dataset.columns, "target_col not found in dataset"
        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.model_type == "electra":
            self.tokenizer = ElectraTokenizer.from_pretrained(
                "google/electra-base-discriminator"
            )
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoded_data = self.tokenizer.batch_encode_plus(
            dataset[feature_col].values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]
        labels = torch.tensor(dataset[target_col].values).long()
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
        return tensor_dataset

    def _dataloaders(self, train_data, val_data, seed):
        # converts tensor dataset into batches
        set_seed(seed)
        train_dataloader = DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            batch_size=self.batch_size,
        )
        validation_dataloader = DataLoader(
            val_data,
            sampler=SequentialSampler(val_data),
            batch_size=self.batch_size,
        )
        return train_dataloader, validation_dataloader

    def _init_model(self):
        # model initialization
        if self.model_type == "bert":
            if not self.int_task:
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    classifier_dropout=self.classifier_dropout,
                )
                print("Bert base initialized and weights loaded..")
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    "ishan/bert-base-uncased-mnli",
                    classifier_dropout=self.classifier_dropout,
                )
                print("Bert base initialized, trained on intermediate task")
        elif self.model_type == "electra":
            if not self.int_task:
                self.model = ElectraForSequenceClassification.from_pretrained(
                    "google/electra-base-discriminator",
                    classifier_dropout=self.classifier_dropout,
                )
                print("Electra base initialized and weights loaded..")
            else:
                self.model = ElectraForSequenceClassification.from_pretrained(
                    "howey/electra-base-mnli",
                    classifier_dropout=self.classifier_dropout,
                )
                print("Electra base initialized, trained on intermediate task")
        else:
            if not self.int_task:
                self.model = RobertaForSequenceClassification.from_pretrained(
                    "roberta-base", classifier_dropout=self.classifier_dropout
                )
                print("Roberta base initialized and weights loaded..")
            else:
                self.model = RobertaForSequenceClassification.from_pretrained(
                    "textattack/roberta-base-MNLI",
                    classifier_dropout=self.classifier_dropout,
                )
                print("Roberta base initialized, trained on intermediate task")

    def _reinit_layer_weights(self):
        # called if self.reinit_layers >0
        if not self.model:
            raise NotImplementedError
        encoder_temp = getattr(
            self.model, self.model_type
        )  # encoder_temp = BertModel / RobertaModel / ElectraModel
        if self.model_type in [
            "bert"
        ]:  # apparently, electra and roberta do not have a pooler
            print("reinitializing pooler...")
            encoder_temp.pooler.dense.weight.data.normal_(
                mean=0.0, std=encoder_temp.config.initializer_range
            )
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        for layer in encoder_temp.encoder.layer[-self.reinit_layers :]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(
                        mean=0.0, std=encoder_temp.config.initializer_range
                    )
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def fit(
        self,
        train_dataset,
        val_dataset,
        feature_col="text",
        target_col="target_orig",
        seed=100,
    ):
        """Fit model weights on training dataset and prints evaluation on 
        validation dataset (accuracy and binary cross entropy loss), according 
        to arguments given in args dict. Save training summary (.xlsx) and 
        training arguments (.json) in a directory './model_summary/'.

        Args:
            train_dataset (pandas dataframe):
                dataset used for training. Contains a text column and a label column.
            val_dataset (pandas dataframe):
                dataset used for validation. Contains a text column and a label column.
            feature_col (str):
                name of the text column, must be identical in train_dataset 
                and val_dataset. Defaults to 'text'.
            target_col (str):
                name of the label column, must be identical in train_dataset 
                and val_dataset. Labels are 0 (no scam) or 1 (scam).
                Defaults to 'target_orig'.
            seed (int, list of int):
                random state for replicable results. Defaults to 100.

        """

        assert isinstance(seed, (int, list))
        if isinstance(seed, int):
            seed = [seed]
        for s in seed:
            assert isinstance(s, int)
            if s < 0:
                raise ValueError

        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        for random_seed in seed:
            train_tensor = self._preprocessing(
                train_dataset, feature_col=feature_col, target_col=target_col
            )
            val_tensor = self._preprocessing(
                val_dataset, feature_col=feature_col, target_col=target_col
            )
            set_seed(random_seed)
            self._init_model()
            if self.reinit_layers:
                if self.reinit_layers > 0:
                    self._reinit_layer_weights()
            self.model.cuda()
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,  
                eps=1e-8, 
            )
            epochs = self.num_epochs
            train_dataloader, val_dataloader = self._dataloaders(
                train_tensor, val_tensor, random_seed
            )
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=round(self.warmup_ratio * total_steps, 0),
                num_training_steps=total_steps,
            )
            for epoch_i in range(0, epochs):
                print("")
                print(
                    "======== Epoch {:} / {:} ========".format(
                        epoch_i + 1, epochs
                    )
                )
                print("Training...")
                total_train_loss = 0
                total_train_accuracy = 0
                self.model.train()
                for step, batch in enumerate(
                    train_dataloader
                ):  # len(train_dataloader) = len(train_dataset)//16
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)
                    self.model.zero_grad()
                    result = self.model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True,
                    )
                    loss = result.loss
                    logits = result.logits
                    total_train_loss += loss.item()
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to("cpu").numpy()
                    total_train_accuracy += flat_accuracy(logits, label_ids)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    optimizer.step()
                    scheduler.step()  # Update the learning rate.
                avg_train_accuracy = total_train_accuracy / len(
                    train_dataloader
                )
                avg_train_loss = total_train_loss / len(train_dataloader)
                print(
                    "  Average training loss: {0:.2f}".format(avg_train_loss)
                )
                print("")
                print("Validation...")
                self.model.eval()
                total_val_accuracy = 0
                total_val_loss = 0
                for batch in val_dataloader:
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)
                    with torch.no_grad():
                        result = self.model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True,
                        )
                    loss = result.loss
                    logits = result.logits
                    total_val_loss += loss.item()
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to("cpu").numpy()
                    total_val_accuracy += flat_accuracy(logits, label_ids)
                avg_val_accuracy = total_val_accuracy / len(val_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
                avg_val_loss = total_val_loss / len(val_dataloader)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            print("")
            print("Training completed")
            if avg_val_loss < self.best_loss:
                self.best_model = self.model
                self.best_loss = avg_val_loss
            self._save_training_summary(
                random_seed,
                avg_train_loss,
                avg_train_accuracy,
                avg_val_loss,
                avg_val_accuracy,
            )

    def _save_training_summary(
        self,
        seed,
        avg_train_loss,
        avg_train_accuracy,
        avg_val_loss,
        avg_val_accuracy,
    ):
        # saves training summary in xlsx and training arguments in .json format in a directory './model_summary/'.
        training_stats = {
            "seed": [seed],
            "training_loss": [avg_train_loss],
            "training_accuracy": [avg_train_accuracy],
            "validation_loss": [avg_val_loss],
            "validation_accuracy": [avg_val_accuracy],
        }
        training_summary = pd.DataFrame(training_stats)
        output_dir = "./model_summary/"
        print("Saving model_summary to %s" % output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            training_summary.to_excel(
                os.path.join(output_dir, "training_summary.xlsx"), index=False
            )
            with open(
                os.path.join(output_dir, "training_arguments.json"), "w"
            ) as file:
                json.dump(self.args, file)
        else:
            df_exist = pd.read_excel(
                os.path.join(output_dir, "training_summary.xlsx")
            )
            df_exist = pd.concat([df_exist, training_summary]).reset_index(
                drop=True
            )
            df_exist.to_excel(
                os.path.join(output_dir, "training_summary.xlsx"), index=False
            )

    def save_best_model(self, output_dir="./model_save/"):
        """Save model and tokenizer with lowest validation loss in directory 
        output_dir.

        Args:
            output_dir (str):
                where model is saved. Created if not existant.
                Defaults to './model_save/'.

        """

        if not self.best_model:
            raise AttributeError("A model needs to be fit.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = (
            self.best_model.module
            if hasattr(self.best_model, "module")
            else self.best_model
        )
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
