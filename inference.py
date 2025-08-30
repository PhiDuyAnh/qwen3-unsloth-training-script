import argparse

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen3-8B")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="Enable thinking tokens or not.")
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_model(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.seq_length,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit
    )
    return model, tokenizer


def invoke_responses(args, model, tokenizer, prompt):
        print("Assistant:")
        message = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking
        )
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu"), 
            max_new_tokens=args.new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            streamer=TextStreamer(tokenizer, skip_prompt=True)
        )


def main(args):
    model, tokenizer = load_model(args)
    print("\n===========================================================")
    print(f"Model {args.model_name} loaded successfully, start chatting!")
    print("===========================================================\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["/exit", "/quit", "/q"]:
            break
        invoke_responses(args, model, tokenizer, user_input)
        print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
