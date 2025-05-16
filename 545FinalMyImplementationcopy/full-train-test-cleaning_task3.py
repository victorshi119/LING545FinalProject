# full-train-test-cleaning_task3.py
# Author: Wenchen (Victor) Shi
# Date: 2025-04-27
# Purpose: Clean and tokenize the full gold-labeled Task 3 data (train, dev, test) for final evaluation.

import os
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Import your cleaning functions (adapt as needed)
from cleaning_task3 import process_line_task3  # Assuming this exists

# === Paths ===
input_folder = "/content/drive/MyDrive/545FinalMyImplementation/gua_spa_train_dev_test_gold"
output_folder = "/content/drive/MyDrive/545FinalMyImplementation/full_tokenized_task3_dataset"
os.makedirs(output_folder, exist_ok=True)

# === Files ===
train_file = os.path.join(input_folder, "gua_spa_train.txt")
dev_file = os.path.join(input_folder, "gua_spa_dev_gold.txt")
test_file = os.path.join(input_folder, "gua_spa_test_gold.txt")

# === Helper Function ===
def load_and_process_file(path):
    examples = []
    tokens = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0].isdigit():
                continue
            token, label = process_line_task3(line)
            tokens.append(token)
            labels.append(label)

    # Group into sentences manually based on "sentence breaks" (assuming each file groups tokens linearly)
    examples.append({
        "tokens": tokens,
        "labels": labels
    })

    return examples

# === Processing ===
print("\U0001F4C1 Processing train...")
train_examples = load_and_process_file(train_file)

print("\U0001F4C1 Processing dev...")
dev_examples = load_and_process_file(dev_file)

print("\U0001F4C1 Processing test...")
test_examples = load_and_process_file(test_file)

# === Create HuggingFace Datasets ===
train_dataset = Dataset.from_list(train_examples)
dev_dataset = Dataset.from_list(dev_examples)
test_dataset = Dataset.from_list(test_examples)

# === Combine ===
full_dataset = DatasetDict({
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_dataset
})

# === Save ===
print("\U0001F4C1 Saving tokenized full dataset...")
full_dataset.save_to_disk(output_folder)

print("✅ Full gold tokenized dataset saved at:", output_folder)

# Now you can load it later with: load_from_disk(output_folder)!