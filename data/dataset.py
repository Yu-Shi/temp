import json
import random

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

