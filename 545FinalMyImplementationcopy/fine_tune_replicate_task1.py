# fine_tune_replicate_task1.py

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ✅ Load tokenized dataset
dataset = DatasetDict.load_from_disk("tokenized_task1_dataset")

# ✅ Labels
label_list = ["es", "gn", "other", "ne-b-per", "ne-i-per", "ne-b-org", "ne-i-org", "ne-b-loc", "ne-i-loc"]
label_to_id = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

# ✅ Model and tokenizer
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# ✅ Data collator with safe padding
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, return_tensors="pt")

# ✅ Evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    all_preds = sum(true_predictions, [])
    all_labels = sum(true_labels, [])

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# ✅ Training configuration
training_args = TrainingArguments(
    output_dir="./gua-spa-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # safe for CPU or MPS
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    remove_unused_columns=False,
    report_to="none"
)

# ✅ Set CPU to avoid MPS crash
device = torch.device("cpu")
model.to(device)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ✅ Train the model
trainer.train()

# ✅ Save final model
trainer.save_model("./gua-spa-model/final")
tokenizer.save_pretrained("./gua-spa-model/final")
print("✅ Training finished and model saved.")



# # fine_tune_replicate_task1.py

# from transformers import (
#     BertTokenizerFast,
#     BertForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForTokenClassification
# )
# from datasets import DatasetDict, load_from_disk
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# # Load tokenized dataset
# dataset: DatasetDict = load_from_disk("./tokenized_task1_dataset")

# # Define label list manually
# label_list = ["es", "gn", "other", "ne-b-per", "ne-i-per", "ne-b-org", "ne-i-org", "ne-b-loc", "ne-i-loc"]
# num_labels = len(label_list)

# # Load tokenizer and model
# model_checkpoint = "bert-base-multilingual-cased"
# tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
# model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# # Compute metrics
# def compute_metrics(p):
#     predictions, labels = p
#     preds = np.argmax(predictions, axis=2)

#     true_predictions = [
#         [label_list[p] for (p, l) in zip(pred, label) if l != -100]
#         for pred, label in zip(preds, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(pred, label) if l != -100]
#         for pred, label in zip(preds, labels)
#     ]

#     all_preds = sum(true_predictions, [])
#     all_labels = sum(true_labels, [])

#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
#     acc = accuracy_score(all_labels, all_preds)

#     return {
#         "accuracy": acc,
#         "f1": f1,
#         "precision": precision,
#         "recall": recall,
#     }

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./gua-spa-model",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=50,
#     remove_unused_columns=False  # Important for token classification
# )

# # Data collator to handle padding dynamically
# data_collator = DataCollatorForTokenClassification(tokenizer)

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["dev"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# # Train and save
# trainer.train()
# trainer.save_model("./gua-spa-model/final")
# tokenizer.save_pretrained("./gua-spa-model/final")

# print("✅ Model fine-tuning complete and saved to ./gua-spa-model/final")
