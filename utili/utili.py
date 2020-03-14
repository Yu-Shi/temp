import numpy as np
import torch
import random

special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
segments_name = ['31-40.jsonlines', '41-50.jsonlines', '51-60.jsonlines', '61-70.jsonlines', '71-80.jsonlines']

MAX_LENGTH = int(100)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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


