# save as: train_mlm_guarani.py

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Load tokenizer and model
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Load a small slice of the dataset for faster training/debugging
dataset = load_dataset("text", data_files={"train": "guarani_wiki.txt"}, split="train[:1000]")

# Tokenization function (short max_length for less memory)
def tokenize(example):
    return tokenizer(
        example["text"],
        return_special_tokens_mask=True,
        truncation=True,
        padding="max_length",
        max_length=64  # Reduced from 512
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# MLM data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mlm_guarani",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Smaller batch for memory
    save_steps=5000,
    save_total_limit=2,
    logging_steps=100,
    report_to="none"  # Disable external logging like wandb
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./mlm_guarani")
tokenizer.save_pretrained("./mlm_guarani")

print("✅ MLM fine-tuning complete on 1,000 Guarani Wikipedia samples.")







# # save as: train_mlm_guarani.py
# from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import load_dataset
# import os

# # Load tokenizer
# model_name = "bert-base-multilingual-cased"
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Load dataset
# # dataset = load_dataset("text", data_files={"train": "guarani_wiki.txt"})
# dataset = load_dataset("text", data_files={"train": "guarani_wiki.txt"}, split="train[:1000]")


# # Tokenization function with memory-safe max_length
# def tokenize(example):
#     return tokenizer(
#         example["text"],
#         return_special_tokens_mask=True,
#         truncation=True,
#         padding="max_length",
#         max_length=64  # ↓ Adjusted from 512 to 64
#     )

# tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# # MLM data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )

# # Load model
# model = BertForMaskedLM.from_pretrained(model_name)

# # Training arguments with smaller batch size
# training_args = TrainingArguments(
#     output_dir="./mlm_guarani",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=1,  # ↓ From 2 to 1 for memory safety
#     save_steps=5000,
#     save_total_limit=2,
#     logging_steps=500,
#     report_to="none"  # optional: disables wandb/logging warnings
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     data_collator=data_collator,
# )

# # Train and save
# trainer.train()
# trainer.save_model("./mlm_guarani")
# tokenizer.save_pretrained("./mlm_guarani")







# from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import load_dataset
# import os

# # Load and tokenize
# model_name = "bert-base-multilingual-cased"
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Prepare dataset
# dataset = load_dataset("text", data_files={"train": "guarani_wiki.txt"})
# def tokenize(example):
#     return tokenizer(example["text"], return_special_tokens_mask=True, truncation=True, max_length=512)
# tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# # Data collator for MLM
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# # Model
# model = BertForMaskedLM.from_pretrained(model_name)

# # Training setup
# training_args = TrainingArguments(
#     output_dir="./mlm_guarani",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     save_steps=10_000,
#     save_total_limit=2,
#     logging_steps=500,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     data_collator=data_collator,
# )

# trainer.train()
# trainer.save_model("./mlm_guarani")
# tokenizer.save_pretrained("./mlm_guarani")