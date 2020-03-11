import shared

from transformers import WEIGHTS_NAME


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(100)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config)), ())


def read_eval_samples(tokenizer, args):
    examples = []
    raw = []

    filename = args.input_file
    with open(filename, encoding="utf-8") as f:
        all_lines = f.readlines()
        size = len(all_lines)
        lines = None
        for line in lines:
            splitted = (line[:-1] if line[-1] == '\n' else line).split('\t')
            queries = splitted[1:]
            topic_number = splitted[0]
            i = 1
            for query in queries[1:]:
                input_sents = queries[:i]
                target_sent = query
                this_example = []
                for sent in input_sents:
                    this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                    this_example.append(tokenizer.sep_token_id)
                this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example.append(tokenizer.bos_token_id)
                begin_pos = len(this_example)
                
                this_example = (topic_number, str(i), this_example, input_sents, target_sent)
                examples.append(this_example)
                i += 1
    
    return examples, raw


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--output_file', type=str, default="predictions.jsonlines",
                        help="Output file")
    parser.add_argument('--input_file', type=str, default="",
                        help="Input file (tsv)")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    examples, raw = read_eval_samples(tokenizer, args)
    output_file = open(args.output_file, 'w', encoding='UTF-8')
    i = 0
    modified_queries = []
    last_topic = "xxx"
    for example in tqdm.tqdm(examples):
        # print(example[0] + "_" + example[1])
        # print("input: " + tokenizer.decode(example[2], clean_up_tokenization_spaces=True))
        if example[0] != last_topic:
            modified_queries.clear()
            last_topic = example[0]

        out = shared.sample_sequence(
            model=model,
            context=example[2],
            length=args.length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
            tokenizer=tokenizer
        )
        # print("output: ", end='')
        out = out[:, len(example[2]):].tolist()
        pred = None
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            # text = text[: ]
            # print(text)
            pred = text

        modified_queries.append(pred.strip())
        first_query = example[3][0]
        if pred.strip().lower() != example[4].strip().lower():
            output_line = json.dumps({"topic_number": example[0], "query_number": example[1], "input": [first_query] + modified_queries, "target": example[4]})
            output_file.write(output_line + '\n')
    
    output_file.close()



if __name__ == '__main__':
    main()
