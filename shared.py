import argparse
import glob
import logging
import os
import random
import json
import shared

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from tqdm import tqdm, trange

from transformers import (AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
segments_name = ['31-40.jsonlines', '41-50.jsonlines', '51-60.jsonlines', '61-70.jsonlines', '71-80.jsonlines']

class ConvSearchExample:
    def __init__(self, topic_number, query_number, ids, labels, pred_begin_pos):
        self.topic_number = topic_number
        self.query_number = query_number
        self.ids = ids
        self.labels = labels
        self.pred_begin_pos = pred_begin_pos
    
    def __repr__(self):
        print('===ConvSearchExample===')
        print(self.topic_number + '_' + self.query_number)
        print('-----------------------')
        print(self.ids)
        print('-----------------------')
        print(self.labels)
        print('-----------------------')
        print(self.pred_begin_pos)
        print('=======================')


def collate_fn(batch_dataset: list):
    return_tuple = [[], [], [], [], []]
    for example in batch_dataset:
        return_tuple[0].append(example.topic_number)
        return_tuple[1].append(example.query_number)
        return_tuple[2].append(example.ids)
        return_tuple[3].append(example.labels)
        return_tuple[4].append(example.pred_begin_pos)
    return_tuple[2] = torch.tensor(return_tuple[2])
    return_tuple[3] = torch.tensor(return_tuple[3])
    return_tuple = tuple(return_tuple)
    return return_tuple


class QueryRewriteDataset(Dataset):
    def __init__(self, filenames, tokenizer, args, num_sampled_sessions):
        self.examples = []
        # filename = args.train_file
        for filename in filenames:
            with open(filenames, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    topic_number = record['topic_number']
                    query_number = record['query_number']
                    this_example = []
                    this_example_labels = []

                    if not args.simplify:

                        for sent in input_sents:
                            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                            this_example.append(tokenizer.sep_token_id)
                        this_example.pop()
                        this_example.append(tokenizer.bos_token_id)

                        begin_pos = len(this_example)
                        this_example_labels.extend([-1] * begin_pos)
                        this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))

                    else:

                        for sent in input_sents[0:len(input_sents)-1]:
                            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                            this_example.append(tokenizer.sep_token_id)
                        this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example.append(tokenizer.bos_token_id)

                        begin_pos = len(this_example)
                        this_example_labels.extend([-1] * begin_pos)
                        this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_sents[-1])))
                        this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_sents[-1])))

                    this_example.append(tokenizer.eos_token_id)
                    this_example_labels.append(tokenizer.eos_token_id)

                    if len(this_example) > args.block_size:
                        this_example = this_example[:args.block_size]
                        this_example_labels = this_example_labels[:args.block_size]
                    else:
                        pad_num = args.block_size - len(this_example)
                        this_example.extend([tokenizer.pad_token_id] * pad_num)
                        this_example_labels.extend([-1] * pad_num)
                    assert len(this_example) == args.block_size
                    assert len(this_example_labels) == args.block_size
                    self.examples.append(shared.ConvSearchExample(topic_number, query_number, this_example, this_example_labels, begin_pos))

        if 0 < num_sampled_sessions < len(topic_numbers):
            sampled_topics = random.choices(list(topic_numbers), k=num_sampled_sessions)
            # print(sampled_topics)
            tmp = []
            for example in self.examples:
                if example.topic_number in sampled_topics:
                    tmp.append(example)
            self.examples = tmp

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def load_examples(filenames, args, tokenizer, num_sampled_sessions=-1):
    dataset = QueryRewriteDataset(filenames, tokenizer, args, num_sampled_sessions)
    return dataset


def train(args, train_dataset, model, tokenizer, cross_validate_id=-1):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=shared.collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch[2], batch[3])  # get ids and labels
            inputs = inputs.to(args.device)  # batch_size * block_size
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            del inputs
            del outputs
            torch.cuda.empty_cache()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            del loss
            torch.cuda.empty_cache()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    segment_output_dir = args.output_dir + (('-' + str(cross_validate_id)) if cross_validate_id != -1 else "")
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, temperature=1, top_p=0.0, repetition_penalty=1.0,
                    device='cpu', tokenizer=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated[0].tolist()):
                next_token_logits[0, _] /= repetition_penalty

            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            new_token = next_token.tolist()
            if tokenizer.decode(new_token[0]) == "<EOS>":
                break
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def read_eval_samples(tokenizer, args, filenames):
    examples = []
    raw = []
    for filename in filenames:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                input_sents = record['input']
                target_sent = record['target']
                topic_number = record['topic_number']
                query_number = record['query_number']
                this_example = []
                
                for sent in input_sents:
                    this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                    this_example.append(tokenizer.sep_token_id)
                this_example.pop()
                this_example.append(tokenizer.bos_token_id)
                begin_pos = len(this_example)
                
                this_example = (topic_number, query_number, this_example, target_sent)
                examples.append(this_example)

                raw.append(record)
    
    return examples, raw