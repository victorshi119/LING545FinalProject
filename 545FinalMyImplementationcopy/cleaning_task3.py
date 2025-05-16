import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_list = ["cc", "ul"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

data_files = {
    "train": "./gua_spa_train_dev_test/gua_spa_train.txt",
    "dev": "./gua_spa_train_dev_test/gua_spa_dev.txt"
}

def load_data(filepath):
    examples = []
    with open(filepath, encoding="utf-8") as f:
        tokens, labels = [], []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens:
                    examples.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
            else:
                try:
                    _, token, lang_tag = line.split()
                    if lang_tag.startswith("es"):
                        label = lang_tag.split("-")[-1]
                        if label in label_to_id:
                            tokens.append(token)
                            labels.append(label_to_id[label])
                except:
                    continue
        if tokens:
            examples.append({"tokens": tokens, "labels": labels})
    return examples

raw_dataset = DatasetDict({
    split: Dataset.from_list(load_data(path))
    for split, path in data_files.items()
})

def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding='max_length', max_length=128)
    word_ids = tokenized.word_ids()

    aligned_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(example["labels"][word_id])
        else:
            aligned_labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = aligned_labels
    return tokenized

tokenized_dataset = raw_dataset.map(tokenize_and_align)
tokenized_dataset.save_to_disk("tokenized_task3_dataset")

print("✅ Task 3 dataset ready and saved to 'tokenized_task3_dataset'")
