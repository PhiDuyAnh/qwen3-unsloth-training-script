import json

from datasets import load_dataset
from unsloth import FastLanguageModel


def generate_conversations(dataset):
    questions = dataset["question"]
    answers = dataset["answer"]
    thoughts = dataset["thought"]
    conversations = []

    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    thinking_start_token = None
    thinking_end_token = None
    for token in tokenizer.get_added_vocab().keys():
        if "think" in token and "/" in token:
            thinking_end_token = token
        elif "think" in token:
            thinking_start_token = token

    for question, answer, thought in zip(questions, answers, thoughts):
        conversations.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": thinking_start_token + thought + thinking_end_token + answer}
        ])
    conversations = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
    )

    return {"conversation": conversations}


def save_dataset(dataset, output_dir):
    with open(output_dir, "w") as f:
        dataset_dict = {"text": dataset["conversation"]}
        json.dump(dataset_dict, f, indent=4)
    print("\n====================================================")
    print(f"Dataset saved to {output_dir} successfully.")
    print("====================================================\n")


if __name__ == "__main__":
    train_dataset = load_dataset(
        "LLMTeamAkiyama/cleand_moremilk_CoT_Reasoning_Quantom_Physics_And_Computing",
        split="train"
    )

    formatted_train_dataset = train_dataset.map(
        generate_conversations,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    save_dataset(formatted_train_dataset, "datasets/example.json")


