
# guarani_mlm_pretrain.py

from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Step 1: Load preprocessed Guarani Wikipedia (plaintext)
data_path = "guarani_wiki.txt"  # Place this file in the same directory or update path

dataset = load_dataset("text", data_files={"train": data_path})

# Step 2: Initialize tokenizer and model
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = BertForMaskedLM.from_pretrained(model_checkpoint)

# Step 3: Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Prepare data collator for masked LM
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="mbert-gn-mlm",
    overwrite_output_dir=True,
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=collator,
    tokenizer=tokenizer
)

# Step 7: Train
trainer.train()

# Save model
trainer.save_model("mbert-gn-mlm")
tokenizer.save_pretrained("mbert-gn-mlm")
print("✅ MLM fine-tuning on Guarani Wikipedia complete!")
