import argparse
import json

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Qwen3-8B",
        help="Pretrained model identifier or local path to load."
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (number of tokens) the model will accept."
    )

    parser.add_argument(
        "--load-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model in 4-bit quantized precision. Use --no-load-4bit to disable."
    )

    parser.add_argument(
        "--load-8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the model in 8-bit quantized precision, a bit more accurate and uses 2x memory."
    )

    parser.add_argument(
        "--full-finetuning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full model finetuning (update all weights). If false, only adapter/LoRA params are trained."
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (r) controlling low-rank decomposition size."
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha value for scaling the adapter updates. Best to choose alpha = rank or rank * 2."
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0,
        help="Dropout probability applied to LoRA adapters (0.0 - 1.0)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size. Increasing leads to more memory usage."
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before performing an optimizer step."
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of scheduler warmup steps."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (ignored if --max-steps is used to stop earlier)."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of training steps. Set to a positive value to limit training regardless of epochs."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate for the optimizer."
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log training metrics every N steps."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_8bit",
        help="Optimizer to use (e.g. 'adagrad', 'adamw_8bit', 'sgd')."
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) coefficient."
    )

    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (e.g. 'linear', 'cosine', 'constant')."
    )

    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Reporting backend for experiment logging (e.g. 'wandb', 'tensorboard', 'none')."
    )
    
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default="datasets/example.json",
        help="Path to the training dataset file (JSON expected in current loader)."
    )

    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=None,
        help="Path to the evaluation dataset JSON file (optional). If omitted, no eval dataset is loaded."
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="lora",
        help="Directory where LoRA weights and other outputs will be saved."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def load_model(args):
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.seq_length,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
        full_finetuning=args.full_finetuning
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None
    )

    print("Model loaded successfully 🤗")
    return model, tokenizer


def load_dataset(data_dir):
    """Load dataset from saved JSON file."""
    try:
        with open(data_dir, "r") as f:
            dataset = json.load(f)
    except Exception as e:
        raise ValueError("Make sure the saved dataset file is in JSON format! Error message:", str(e))
    
    text_dataset = Dataset.from_dict(dataset)
    cols = text_dataset.column_names
    if cols[0] != "text":
        text_dataset = text_dataset.rename_column(cols[0], "text")

    print(f"Dataset loaded from {data_dir}.")
    return text_dataset


def main(args):
    """Main training script."""
    # Load model and tokenizer
    model, tokenizer = load_model(args)

    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler_type,
        seed=args.seed,
        report_to=args.report_to
    )

    # Load training, eval datasets
    train_dataset = load_dataset(args.train_data_dir)

    if args.eval_data_dir: # Optional
        eval_dataset = load_dataset(args.eval_data_dir)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            args=training_args
        )

    # Start training!
    trainer_stats = trainer.train()
    
    # Save trained model
    model.save_pretrained(args.save_dir)

    return trainer_stats


if __name__ == "__main__":
    args = parse_args()
    trainer_stats = main(args)