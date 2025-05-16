import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re

# Define file paths
data_files = {
    "train": "./gua_spa_train_dev_test/gua_spa_train.txt",
    "dev": "./gua_spa_train_dev_test/gua_spa_dev_gold.txt",
    "test": "./gua_spa_train_dev_test/gua_spa_test.txt",  # Unlabeled
}

# Load tokenizer
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Label list and mappings (excluding test labels)
label_list = [
    "es", "gn", "other",
    "ne-b-per", "ne-i-per",
    "ne-b-org", "ne-i-org",
    "ne-b-loc", "ne-i-loc"
]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

def load_data(split):
    examples = []
    tokens, labels = [], []

    with open(data_files[split], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens:
                    # Append only tokens if test set
                    if split == "test":
                        examples.append({"tokens": tokens})
                    else:
                        examples.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
            else:
                parts = line.split()
                if split == "test":
                    if len(parts) >= 2:
                        tokens.append(parts[1])
                else:
                    if len(parts) == 3:
                        _, token, label = parts
                        tokens.append(token)
                        labels.append(label_to_id.get(label, label_to_id["other"]))
                    else:
                        continue  # skip malformed lines

        # Catch final block
        if tokens:
            if split == "test":
                examples.append({"tokens": tokens})
            else:
                examples.append({"tokens": tokens, "labels": labels})

    print(f"✅ Loaded {len(examples)} examples for split '{split}'")
    return examples


# Load each split
raw_dataset = DatasetDict({
    split: Dataset.from_list(load_data(split))
    for split in ["train", "dev", "test"]
})


# Tokenization + alignment
def tokenize_and_align(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()

    if "labels" in example:
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(example["labels"][word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        tokenized["labels"] = aligned_labels

    return tokenized


print("🔄 Tokenizing...")
tokenized_dataset = raw_dataset.map(tokenize_and_align, remove_columns=["tokens"])

# Save dataset
tokenized_dataset.save_to_disk("tokenized_task1_dataset")
print("✅ Tokenized dataset saved to: tokenized_task1_dataset")
