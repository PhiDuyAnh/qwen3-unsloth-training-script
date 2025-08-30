# 🦥 Qwen3 - Unsloth Supervised Fine-Tuning Script

Easily train Qwen3 models using Supervised Fine-Tuning (SFT) with Unsloth and run inference on your local computer!

## ⚙️ Dependencies

Make sure Unsloth is installed:
```bash
pip install -r requirements.txt
# or just
pip install unsloth
```
## 🔧 How to start training

1. Prepare your dataset in JSON format. Refer to `datasets/example.json` for the correct format and `datasets/example.py` to look at how to prepare one for yourself.

2. Run `train.py`.

Example:
```bash
python train.py \
--model-name="unsloth/Qwen3-4B" \
--lora-rank=16 \
--lora-alpha=16 \
--lora-dropout=0.1 \
--seed=1024 \
--batch-size=4 \
--warmup-steps=0 \
--epochs=10 \
--max-steps=-1 \
--lr=5e-4 \
--logging-steps=30 \
--train-data-dir=datasets/example.json \
--save-dir=qwen3_4b_lora
```

## 🚀 Running inference

You can run inference and chat with your models both before and after training. Input your saved model/LoRA directory for the `--model-name` argunment.

Example:
```bash
python inference.py \
--model-name="lora" \
--seq-length=4096 \
--no-thinking \
--new-tokens=2048 \
--temperature=0.7 \
--top-p=0.8 \
--top-k=20
```

Type '/exit', '/quit' or '/q' to exit chatting.
