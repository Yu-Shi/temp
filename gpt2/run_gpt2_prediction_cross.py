import argparse
from utili import utili
import logging
from training import train
from data.dataset import read_eval_samples
import torch
from transformers import (WEIGHTS_NAME, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from prediction import sample_sequence
from utili import MAX_LENGTH

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
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
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    for i in range(len(utili.segments_name)):

        eval_segment = utili.segments_name[i]

        utili.set_seed(args)

        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = tokenizer_class.from_pretrained(f"{args.model_name_or_path}-{i}")
        model = model_class.from_pretrained(f"{args.model_name_or_path}-{i}")
        print(f"Loading model {args.model_name_or_path}-{i}")
        model.to(args.device)
        model.eval()


        if args.length < 0 and model.config.max_position_embeddings > 0:
            args.length = model.config.max_position_embeddings
        elif 0 < model.config.max_position_embeddings < args.length:
            args.length = model.config.max_position_embeddings  # No generation bigger than model size 
        elif args.length < 0:
            args.length = MAX_LENGTH  # avoid infinite loop

        examples, raw = read_eval_samples(tokenizer, args, [eval_segment])
        i = 0
        for example in examples:
            out = sample_sequence(
                model=model,
                context=example[2],
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
                tokenizer=tokenizer
            )
            out = out[:, len(example[2]):].tolist()
            pred = None
            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                pred = text
            raw[i]['output'] = pred
            output_line = json.dumps(raw[i])
            i += 1
            output_file.write(output_line + '\n')
        
        del model
        torch.cuda.empty_cache() 
        
    output_file.close()

if __name__ == '__main__':
    main()
