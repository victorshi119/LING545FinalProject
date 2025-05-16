from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from datasets import DatasetDict, load_from_disk
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

label_list = ["cc", "ul"]
num_labels = len(label_list)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)

dataset = load_from_disk("tokenized_task3_dataset")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_preds = [[label_list[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                  for pred_row, label_row in zip(preds, p.label_ids)]
    true_labels = [[label_list[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                   for pred_row, label_row in zip(preds, p.label_ids)]

    all_preds = sum(true_preds, [])
    all_labels = sum(true_labels, [])

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./task3-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./task3-model/final")
print("✅ Task 3 fine-tuning complete!")
