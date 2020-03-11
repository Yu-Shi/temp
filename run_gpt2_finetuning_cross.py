import shared

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--load_from_multiple_models", action='store_true', 
                        help="Whether to load from multiple models with postfix -0 to -4.")
    parser.add_argument("--block_size", default=150, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--start_checkpoint", default=-1, type=int, help="Start from checkpoint X.")
    parser.add_argument("--num_sampled_sessions", default=40, type=int,
                        help="Sample X sessions in a total of 40.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--simplify", action="store_true", 
                        help="Set this flag to train the simplifier, not the rewriter.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Training
    for i in range(0, len(shared.segments_name)):

        # Set seed
        set_seed(args)

        config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        postfix = ('-' + str(i)) if args.load_from_multiple_models else ''
        
        config = config_class.from_pretrained(args.model_name_or_path + postfix)  # gpt2 size

        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + postfix)
        tokenizer.add_special_tokens(shared.special_tokens_dict)

        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

        postfix += ('/checkpoint-' + str(args.start_checkpoint)) if (args.start_checkpoint != -1) else ''
        model = model_class.from_pretrained(args.model_name_or_path + postfix)
        model.resize_token_embeddings(len(tokenizer))  # resize
        assert tokenizer.sep_token == '<SEP>'
        assert tokenizer.pad_token == '<PAD>'
        assert tokenizer.bos_token == '<BOS>'
        assert tokenizer.eos_token == "<EOS>"
        logger.info("Added sep_token (id: %s), pad_token (id: %s), bos_token (id: %s) and eos_token (id: %s)", tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
        model.to(args.device)

        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

        logger.info("Training/evaluation parameters %s", args)

        train_segments = [shared.segments_name[x] for x in range(i)] + [shared.segments_name[x] for x in range(i+1, len(shared.segments_name))]
        print("train_segments: {}".format(train_segments))

        train_dataset = shared.load_examples(train_segments, args, tokenizer, args.num_sampled_sesions)
        global_step, tr_loss = shared.train(args, train_dataset, model, tokenizer, cross_validate_id=i)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        # Create output directory if needed
        segment_output_dir = args.output_dir + '-' + str(i)
        if not os.path.exists(segment_output_dir):
            os.makedirs(segment_output_dir)

        logger.info("Saving model checkpoint to %s", segment_output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(segment_output_dir)
        tokenizer.save_pretrained(segment_output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(segment_output_dir, 'training_args.bin'))

        del model  # free gpu memory
        torch.cuda.empty_cache() 


if __name__ == "__main__":
    main()
