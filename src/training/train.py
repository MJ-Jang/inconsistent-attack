# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import argparse
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from operators import TrainOperator
from utils import ReverseDataset


def return_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str,  choices=['esnli', 'cose1.0'],
                        default='esnli',)
    parser.add_argument('--task_type', type=str,  choices=['reverse'],
                        default='reverse',
                        help='type of task')
    parser.add_argument('--data_directory', type=str, default='../../resources/esnli_sample',
                        help='directory where training/dev/test datasets are located')
    parser.add_argument('--save_dir', type=str, default='../../model_binary',
                        help='directory where trained model binary will be placed')

    # Others
    parser.add_argument('--model_type', type=str, default='t5-base',
                        help='type of T5 model, ex) t5-base')
    parser.add_argument('--saved_model_path', type=str, default=None,
                        help="previously trained model path")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--logging_steps', type=int, default=10000,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--no_teacher_force', action='store_true',
                        help='Avoid using teacher-force training process')
    ## for training
    parser.add_argument('--patience', type=int, default=3,
                        help='number of patience for early stopping')
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    return parser.parse_args()


def return_dataset(args, tokenizer, data_type):
    assert data_type in ['train', 'test', 'dev']
    return ReverseDataset(tokenizer, args, directory=args.data_directory, data_type=data_type)


def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # Load pretrained model and tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(args.model_type)
    model = T5ForConditionalGeneration.from_pretrained(args.model_type, return_dict=True)
    model.to(device)

    # Load dataset
    train_dataset = return_dataset(args, tokenizer, data_type='train')
    eval_dataset = return_dataset(args, tokenizer, data_type='dev')

    # Declare Trainer
    kwargs = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }

    trainer = TrainOperator(**kwargs)

    # Run train
    save_prefix = f"{args.model_type}-{args.task_type}-{args.dataset}"
    trainer.run_train(args, save_prefix=save_prefix)


if __name__ == '__main__':
    args = return_args()
    main(args)

