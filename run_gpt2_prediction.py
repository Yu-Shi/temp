import shared

from transformers import WEIGHTS_NAME


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(100)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config)), ())


def read_eval_samples(tokenizer: GPT2Tokenizer, args):
    file_paths = shared.segments_name

    examples = []
    raw = []
    for filename in file_paths:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                input_sents = record['input']
                target_sent = record.get("target", "")
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


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--output_file', type=str, default="predictions",
                        help="Output file for predictions")
    parser.add_argument('--eval_all_checkpoints', action='store_true', help="Predict from model of all checkpoints.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    checkpoints = [args.model_name_or_path]
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.model_name_or_path + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:

        global_step = ('-' + checkpoint.split('-')[-1]) if checkpoint.split('-')[-1].isdigit() else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        model.eval()

        if args.length < 0 and model.config.max_position_embeddings > 0:
            args.length = model.config.max_position_embeddings
        elif 0 < model.config.max_position_embeddings < args.length:
            args.length = model.config.max_position_embeddings  # No generation bigger than model size 
        elif args.length < 0:
            args.length = MAX_LENGTH  # avoid infinite loop

        examples, raw = shared.read_eval_samples(tokenizer, args, shared.segments_name)
        output_file = open(args.output_file + global_step + '.jsonlines', 'w', encoding='UTF-8')
        i = 0
        for example in examples:
            # print(example[0] + "_" + example[1])
            # print("input: " + tokenizer.decode(example[2], clean_up_tokenization_spaces=True))
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
                # print(text)
                pred = text
            # print("target: " + example[3])
            # print()
            raw[i]['output'] = pred
            output_line = json.dumps(raw[i])
            i += 1
            output_file.write(output_line + '\n')
        
        output_file.close()
        if not args.eval_all_checkpoints:
            break

if __name__ == '__main__':
    main()
