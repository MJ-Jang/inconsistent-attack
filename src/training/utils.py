# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import logging
import logging.config
import sys
import numpy as np
import torch
import pickle
import abc
import os
import pandas as pd
import json

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Text


# ================== Logger ================================
def Logger(file_name):
    formatter = logging.Formatter(fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')  # %I:%M:%S %p AM|PM format
    logging.basicConfig(filename='%s.log' % (file_name),
                        format='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S', filemode='w', level=logging.INFO)
    log_obj = logging.getLogger()
    log_obj.setLevel(logging.INFO)
    # log_obj = logging.getLogger().addHandler(logging.StreamHandler())

    # console printer
    screen_handler = logging.StreamHandler(stream=sys.stdout)  # stream=sys.stdout is similar to normal print
    screen_handler.setFormatter(formatter)
    logging.getLogger().addHandler(screen_handler)

    log_obj.info("Logger object created successfully..")
    return log_obj


# ================== EarlyStopping ================================
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=4, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, n_gpu=1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.n_gpu = n_gpu

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.n_gpu <= 1:
            torch.save(model.state_dict(), self.path)
        else:
            torch.save(model.module.state_dict(), self.path)
        self.val_loss_min = val_loss


class BaseTrainDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, tokenizer, args, directory='train', data_type='train',
                 save_cache: bool = True, use_cache: bool = True):
        """
        Base Dataset Class.
        For a new dataset, one should implement 'input_process' and 'target_process' functions

        Args:
            tokenizer: Tokenizer
            args: args should have 'train_ratio', 'seed' and 'task_type'
            directory: directory to fetch data
            data_type: 'train', 'dev', or 'test'
        """
        self.eos_token_id = tokenizer.eos_token_id
        self.cross_entropy_ignore_index = -100
        self.save_cache = save_cache
        self.tokenizer = tokenizer

        cached_features_file, data = self.load_data(directory, data_type, args)
        self.data = data

        if use_cache:
            if os.path.exists(cached_features_file) and data_type in ['train']:
                print('Loading features from', cached_features_file)
                with open(cached_features_file, 'rb') as handle:
                    self.examples = pickle.load(handle)
                return

        self.examples = self.create_example(self.data)

        if data_type == 'train':
            if self.save_cache:
                print('Saving ', len(self.examples), ' examples')
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.examples.items()}
        if 'labels' in list(item.keys()):
            labels = item['labels']
            prompt_length = int(sum(labels != 0))
            item['labels'][prompt_length:] = -100
        return item

    def create_example(self, data):
        inputs = data['input'].tolist()
        labels = data['label'].tolist()
        if 'expl' in list(data.keys()):
            expls = data['expl'].tolist()
        else:
            expls = []

        inputs, targets = self.process(inputs, expls, labels)
        print(f"Input {inputs[0]} | Target: {targets[0]}")
        del labels
        del expls

        input_encode = self.tokenizer(inputs, truncation=True, padding=True)
        del inputs

        print('Encoding inputs')
        target_encode = self.tokenizer(targets, truncation=True, padding=True)
        del targets
        print('Encoding targets')

        outputs = {
            "labels": target_encode['input_ids']
        }
        del target_encode

        outputs["input_ids"] = input_encode['input_ids']
        outputs["attention_mask"] = input_encode['attention_mask']
        del input_encode

        print("Finished preprocess")
        return outputs

    @staticmethod
    def load_data(directory: Text, data_type: Text, args):
        print(f"Load data from {os.path.join(directory, f'{data_type}.tsv')}")
        data_path = os.path.join(directory, f'{data_type}.tsv')
        assert os.path.isfile(data_path)

        data = pd.read_csv(data_path, sep='\t')
        cached_features_file = os.path.join(directory, f'cached_{args.task_type}_{data_type}')
        return cached_features_file, data

    @abc.abstractmethod
    def input_process(self, sent: Text = None, label: Text = None) -> Text:
        pass

    @abc.abstractmethod
    def target_process(self, sent: Text = None, label: Text = None) -> Text:
        pass

    @abc.abstractmethod
    def process(self, inputs, expls, labels):
        pass


class ReverseDataset(BaseTrainDataset):

    @staticmethod
    def load_data(directory: Text, data_type: Text, args):
        print(f"Load data from {os.path.join(directory, f'reverse_{data_type}.tsv')}")
        data_path = os.path.join(directory, f'reverse_{data_type}.tsv')
        assert os.path.isfile(data_path)

        data = pd.read_csv(data_path, sep='\t', index_col='pairID')
        cached_features_file = os.path.join(directory, f'cached_reverse_{data_type}')
        return cached_features_file, data

    def input_process(self, sent: Text = None, label: Text = None) -> Text:
        return f"reverse {sent}"

    def target_process(self, sent: Text = None, label: Text = None) -> Text:
        return f"{sent}"

    def process(self, inputs, expls, labels):
        inputs = [self.input_process(sent=s) for s in tqdm(inputs, desc='input processing')]
        targets = [self.target_process(sent=s) for s
                   in tqdm(labels, desc='target processing', total=len(expls))]
        return inputs, targets


class InconsistVarGenDataset(BaseTrainDataset):

    @staticmethod
    def load_data(directory: Text, data_type: Text, args):
        print(f"Load data from {os.path.join(directory, f'inconsist-expls-test.json')}")
        data_path = os.path.join(directory, f'inconsist-expls-test.json')
        assert os.path.isfile(data_path)

        cached_features_file = ""
        with open(data_path, 'r', encoding='utf-8') as loadFile:
            data = json.load(loadFile)
        return cached_features_file, data

    def input_process(self, sent: Text = None, label: Text = None) -> Text:
        return f"reverse {sent}"

    def target_process(self, sent: Text = None, label: Text = None) -> Text:
        pass

    def process(self, inputs, expls, label):
        sents_ = [f"{c} Explanation: {e}" for c,e in zip(inputs, expls)]
        inputs = [self.input_process(sent=s) for s in tqdm(sents_, desc='input processing')]
        return inputs

    def create_example(self, data):
        context = data['context']
        expls = data['inconsist_expl']

        inputs = self.process(context, expls, None)
        print(f"Input {inputs[0]}")
        del expls

        input_encode = self.tokenizer(inputs, truncation=True, padding=True)
        del inputs
        print('Encoding inputs')

        outputs = dict()
        outputs["input_ids"] = input_encode['input_ids']
        outputs["attention_mask"] = input_encode['attention_mask']
        del input_encode

        print("Finished preprocess")
        return outputs


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_type', type=str, default='reverse')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = ReverseDataset(
        tokenizer=tokenizer,
        args=args,
        directory='../../resources/esnli',
        data_type='test'
    )
    item = dataset.__getitem__(0)
    print(item)

    dataset = InconsistVarGenDataset(
        tokenizer=tokenizer,
        args=args,
        directory='../../resources/esnli',
        data_type='test'
    )
    item = dataset.__getitem__(0)
    print(item)