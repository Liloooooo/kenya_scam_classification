"""Module scam_evaluator -- class ScamEvaluator: evaluate data, predict labels/probabilities, and get Shapley values on loaded Bert-like pytorch model."""


import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
    BertTokenizer,
    ElectraTokenizer,
    RobertaTokenizer,
    BasicTokenizer,
)
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import json
from shap.maskers._text import Text
from shap.utils.transformers import getattr_silent
import shap
import scipy as sp


class ScamEvaluator:
    """Evaluation of data given Bert-like Pytorch model.

    Attrs:
        model_path (str): path to model, saved with Huggingface's .save_pretrained() method.

    Methods:
        evaluate(test_dataset, feature_col='text', target_col='target_orig'):
            returns and prints loss and accuracy of text/label combinations.
            Takes in a pandas dataframe.
        predict_proba(test_dataset, feature_col='text'):
            returns predicted probability array. Takes in a pandas dataframe.
        predict_labels(test_dataset, feature_col='text'):
            returns predicted labels array. Takes in a pandas dataframe.
        shap_values(data): 
            returns shap Explanation object, containing coalitional Shapley values of tokens 
            extracted by transformers.BasicTokenizer.  

    """

    def __init__(self, model_path):
        self.model_path = model_path
        with open(self.model_path + "config.json") as config_json:
            config_dict = json.load(config_json)
        self.model_type = config_dict["model_type"]
        if not self.model_type in ["bert", "electra", "roberta"]:
            raise KeyError("Only Bert, Electra and Roberta models supported.")
        if self.model_type == "electra":
            self.model = ElectraForSequenceClassification.from_pretrained(
                self.model_path
            )
            print("Fine-tuned Electra model loaded")
        elif self.model_type == "bert":
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path
            )
            ("Fine-tuned Bert model loaded")
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_path
            )
            ("Fine-tuned Roberta model loaded")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _preprocessing(
        self,
        test_dataset,
        feature_col="text",
        target_col="target_orig",
        with_labels=True,
    ):
        # tokenizes text in pandas dataframe, list, series or array and returns 
        #a tensor dataset with columns input_ids, attention_mask and labels (if with_labels)
        if with_labels:
            assert isinstance(
                test_dataset, pd.DataFrame
            ), "dataset must be a pandas DataFrame."
            assert (
                target_col in test_dataset.columns
            ), "target_col not found in dataset"
        else:
            assert isinstance(
                test_dataset, (list, pd.DataFrame, pd.Series, np.ndarray)
            ), "input must be pandas DataFrame, pandas Series or list"
            test_dataset = pd.DataFrame(test_dataset, columns=[feature_col])
        assert (
            feature_col in test_dataset.columns
        ), "feature_col not found in dataset"
        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.model_type == "electra":
            self.tokenizer = ElectraTokenizer.from_pretrained(
                "google/electra-base-discriminator"
            )
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        encoded_data = self.tokenizer.batch_encode_plus(
            test_dataset[feature_col].values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]
        if with_labels:
            labels = torch.tensor(test_dataset[target_col].values).long()
            tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
        else:
            tensor_dataset = TensorDataset(input_ids, attention_masks)
        return tensor_dataset

    def _dataloader(self, test_data):
        # converts tensor dataset into batches
        test_dataloader = DataLoader(
            test_data,  # tensor dataset
            sampler=SequentialSampler(test_data),
            batch_size=32,
        )
        return test_dataloader

    def evaluate(
        self, test_dataset, feature_col="text", target_col="target_orig"
    ):
        """Evaluate and print loss and accuracy for a labeled pandas dataframe.

        Args:
            test_dataset (pandas dataframe):
                dataset to be evaluated. Contains a text column and a label column.
            feature_col (str, optional):
                name of the text column. Defaults to 'text'
            target_col (str, optional):
                name of the label column. Labels are 0 (no scam) or 1 (scam).
                Defaults to 'target_orig'.

        Returns:
            loss (float), accuracy (float)

        """

        test_tensor = self._preprocessing(
            test_dataset,
            feature_col=feature_col,
            target_col=target_col,
            with_labels=True,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        test_dataloader = self._dataloader(test_tensor)

        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        self.model.eval()
        total_test_accuracy = 0
        total_test_loss = 0
        for batch in test_dataloader:
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
            total_test_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            total_test_accuracy += flat_accuracy(logits, label_ids)
        avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
        avg_test_loss = total_test_loss / len(test_dataloader)
        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("")
        return avg_test_loss, avg_test_accuracy

    def _predict(self, test_data, feature_col="text"):
        # processing and forward propagation; returns predicted_probabilities 
        # as numpy array of shape (n, 2)
        predict_tensor = self._preprocessing(
            test_data, feature_col=feature_col, with_labels=False
        )
        if torch.cuda.is_available():
            self.model.cuda()
        predict_dataloader = self._dataloader(predict_tensor)
        self.model.eval()
        probs = []
        for batch in predict_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            with torch.no_grad():
                result = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )
                logits = result.logits.detach().cpu().numpy()
                probas = sp.special.expit(logits)
                probs += list(probas)
        return np.array(probs)

    def predict_proba(self, test_data, feature_col="text"):
        """Predict probabilities for scam.

        Args:
            test_dataset (pandas dataframe, pandas series or list of strings):
                text data to be predicted.
            feature_col (str, optional):
                name of the text column if test_data is a dataframe. Defaults to 'text'.

        Returns:
            predicted_probabilities (numpy array)

        """

        return self._predict(test_data, feature_col=feature_col)[:, 1]

    def predict_labels(self, test_dataset, feature_col="text"):
        """Predict labels.

        Args:
            test_dataset (pandas dataframe):
                dataset to be evaluated. Contains a text column.
            feature_col (str, optional):
                name of the text column. Defaults to 'text'.

        Returns:
            predicted_labels (numpy array)

        """

        return np.round(
            self._predict(test_dataset, feature_col=feature_col)[:, 1]
        )

    def shap_values(self, data):
        """Calculate coalitional Shapley values

        Args:
            data (pandas series, np.array or list): messages to be analyzed

        Returns:
            coalitional Shapley values (shap._explanation.Explanation object)

        """

        class TextMasker(Text):
            def __init__(
                self,
                tokenizer=None,
                mask_token=None,
                collapse_mask_token="auto",
                output_type="string",
            ):
                super().__init__(
                    tokenizer=tokenizer,
                    mask_token=mask_token,
                    collapse_mask_token=collapse_mask_token,
                    output_type=output_type,
                )
                if mask_token is None:
                    if getattr_silent(self.tokenizer, "mask_token") is None:
                        self.mask_token = ""

        masker = TextMasker(self._custom_tokenizer)
        explainer = shap.Explainer(
            self._predict, masker, algorithm="partition"
        )
        return explainer(data)

    def _custom_tokenizer(self, text, return_offsets_mapping=True):
        # tokenizes messages according to BasicTokenizer from transformers library
        text = text.lower()
        wordpunct = BasicTokenizer()
        splitted_text_list = wordpunct.tokenize(text)
        pos_list = []
        pos = 0
        for item in splitted_text_list:
            start = text.find(item, pos)
            end = start + len(item)
            pos_list.append((start, end))
            pos = end
        out = {}
        out["input_ids"] = splitted_text_list
        out["offset_mapping"] = pos_list
        return out
