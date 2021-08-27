# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import argparse
import os
import torch
import json

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from operators import BaseInferenceOperator
from utils import InconsistVarGenDataset
from typing import List, Dict, Text

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def return_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str,  choices=['esnli', 'cose1.0'],
                        default='esnli',)
    parser.add_argument('--data_directory', type=str, default='../../resources/esnli',
                        help='directory where training/dev/test datasets are located')
    parser.add_argument('--save_dir', type=str, default='../../resources/esnli',
                        help='directory where trained model binary will be placed')
    parser.add_argument('--data_type', type=str, default='test',
                        help='type of data going to be used for inference: test/dev')

    # Others
    parser.add_argument('--model_directory', type=str, default='../../model_binary',
                        help='directory path where model binary files are located')
    parser.add_argument('--model_type', type=str, default='t5-base',
                        help='type of T5 model, ex) t5-base')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    return parser.parse_args()


def load_model(args):
    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    model = T5ForConditionalGeneration.from_pretrained(args.model_type, return_dict=True)

    model_prefix = f"{args.model_type}-reverse-{args.dataset}.model"
    try:
        model_path = os.path.join(args.model_directory, model_prefix)
        print(f"load model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise FileNotFoundError("No model binary exist")
    return tokenizer, model


def postprocess(task_type: Text, sent: Text):
    if task_type == 'original' or task_type == 'inconsist_expl' or task_type == 'inconsist_extract':
        if 'explanation' in sent:
            if len(sent.split('explanation')) > 2:
                label = sent.split('explanation')[0]
                expl = ' '.join(sent.split('explanation'))[1:]
            else:
                label, expl = sent.split('explanation')
            label = label.strip()
            expl = expl.strip()
        else:
            label = sent
            expl = ''
        return label, expl
    else:
        raise NotImplementedError


def main(args):
    os.makedirs(os.path.join(args.save_dir, args.model_type), exist_ok=True)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    tokenizer, model = load_model(args)
    model.to(device)
    model.eval()

    eval_dataset = InconsistVarGenDataset(
        tokenizer=tokenizer,
        args=args,
        directory=args.data_directory,
        data_type=args.data_type
    )

    kwargs = {
        "eval_dataset": eval_dataset,
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }

    inferencer = BaseInferenceOperator(**kwargs)
    # Generate
    variables = inferencer.generate(args)

    # Save results
    # 1) load original inconsist-expls
    with open(os.path.join(args.data_directory, f'inconsist-expls-{args.data_type}.json'), 'r', encoding='utf-8') as loadFile:
        data = json.load(loadFile)

    new_outp = {
        'pair_id': data['pair_id'],
        "context": data['context'],
        "variable": variables,
        "tags": data['tags']
    }

    save_filename = f'inconsist-variables-{args.data_type}.json'
    with open(os.path.join(args.save_dir, save_filename), 'w', encoding='utf-8') as saveFile:
        json.dump(new_outp, saveFile)


if __name__ == '__main__':
    args = return_args()
    main(args)
