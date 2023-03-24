# -*- coding: utf-8 -*-

__license__ = "Apache License"
__version__ = "2021.1"
__date__ = "25 08 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import torch
import random
import re
import numpy as np
import os

from transformers import AdamW, get_linear_schedule_with_warmup, AdamWeightDecay
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm
from typing import List, Text, Dict
from utils import Logger, EarlyStopping


class TrainOperator:

    def __init__(self, **kwargs):
        self._train_dataset = kwargs['train_dataset']
        self._eval_dataset = kwargs['eval_dataset']
        self._model = kwargs['model']
        self._tokenizer = kwargs['tokenizer']
        self._device = kwargs['device']

    @staticmethod
    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    @staticmethod
    def save_state_dict(model, save_path: str, save_prefix: str, args):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, save_prefix + '.model')
        model = model.cpu()
        if args.n_gpu <= 1:
            torch.save(model.state_dict(), filename)
        else:
            torch.save(model.module.state_dict(), filename)

    def run_train(self, args, save_prefix):

        # Setup logger and early stopper
        check_path = f'check-{args.model_type}-{args.task_type}.pt'
        logger_name = f'logger-{args.model_type}-{args.task_type}-{args.dataset}'

        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=check_path)
        logger = Logger(logger_name)

        """ Train the model """
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(self._train_dataset) if args.local_rank == -1 else DistributedSampler(self._train_dataset)
        train_dataloader = DataLoader(self._train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self._train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self._model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        self.set_seed(args)

        for e in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                self._model.train()

                max_input_len = torch.sum(batch['input_ids'] != 0, dim=1).max().item()

                inputs = dict()
                inputs['input_ids'] = batch['input_ids'][:, :max_input_len].to(self._device)
                inputs['attention_mask'] = batch['attention_mask'][:, :max_input_len].to(self._device)
                inputs['labels'] = batch['labels'].to(self._device)

                outputs = self._model(**inputs)
                loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self._model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        results = self.run_evaluate(args, self._model, self._device)
                        eval_loss, eval_ppl = results['loss'], results['perplexity']

                        early_stopping(eval_loss, self._model)
                        logger.info(
                            f"Global step {global_step} | Validation loss: {eval_loss} | Validation ppl: {eval_ppl}")

                if (args.max_steps > 0 and global_step > args.max_steps) or early_stopping.early_stop:
                    epoch_iterator.close()
                    break

            if (args.max_steps > 0 and global_step > args.max_steps) or early_stopping.early_stop:
                train_iterator.close()
                break

        self._model.load_state_dict(torch.load(check_path))
        self.save_state_dict(self._model, args.save_dir, save_prefix, args)
        os.remove(check_path)

        logger.info(" global_step = %s, tr average loss = %s", global_step, tr_loss / global_step)
        return global_step, tr_loss / global_step

    def run_evaluate(self, args, model, device):
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(self._eval_dataset) if args.local_rank == -1 else DistributedSampler(self._eval_dataset)
        eval_dataloader = DataLoader(self._eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            max_input_len = torch.sum(batch['input_ids'] != 0, dim=1).max().item()

            inputs = dict()
            inputs['input_ids'] = batch['input_ids'][:, :max_input_len].to(device)
            inputs['attention_mask'] = batch['attention_mask'][:, :max_input_len].to(device)
            inputs['labels'] = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                eval_loss += loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {
            "loss": eval_loss,
            "perplexity": perplexity
        }
        return result


class BaseInferenceOperator:

    def __init__(self, **kwargs):
        self._eval_dataset = kwargs['eval_dataset']
        self._model = kwargs['model']
        self._tokenizer = kwargs['tokenizer']
        self._device = kwargs['device']

        self.PREPRO_PATTERN = re.compile('<[/a-zA-Z]+>')

    def prepro_generated_sent(self, sent: Text) -> Text:
        return self.PREPRO_PATTERN.sub(repl='', string=sent).strip()

    def generate(self, args) -> List:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(self._eval_dataset) if args.local_rank == -1 else DistributedSampler(self._eval_dataset)
        eval_dataloader = DataLoader(self._eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        outputs = []
        for batch in tqdm(eval_dataloader, desc="Generating"):
            max_input_len = torch.sum(batch['input_ids'] != 0, dim=1).max().item()

            inputs = dict()
            inputs['input_ids'] = batch['input_ids'][:, :max_input_len].to(self._device)
            inputs['attention_mask'] = batch['attention_mask'][:, :max_input_len].to(self._device)
            inputs['max_length'] = 100

            with torch.no_grad():
                outp_ = self._model.generate(** inputs,
                                             num_beams=5,
                                             repetition_penalty=2.5,
                                             length_penalty=1.0,
                                             early_stopping=True
                                             )
                for o in outp_:
                    generated = self._tokenizer.decode(o, max_length=100)
                    generated = self.prepro_generated_sent(generated)
                    outputs.append(generated)
        return outputs
