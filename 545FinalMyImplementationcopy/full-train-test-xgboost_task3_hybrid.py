# full train-test-xgboost_task3_hybrid.py
# #🚀 Full Test Evaluation Pipeline: Training + Prediction + Scoring

import os
import numpy as np
import pandas as pd
from datasets import load_from_disk, concatenate_datasets
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
from tqdm import tqdm

# =========================
# 1. Load Datasets
# =========================
print("\U0001F4C1 Loading tokenized dataset...")
dataset_path = "/content/drive/MyDrive/545FinalMyImplementation/tokenized_task3_dataset"
dataset = load_from_disk(dataset_path)

train_dev = concatenate_datasets([dataset["train"], dataset["dev"]])
test_set = dataset["test"]

# =========================
# 2. Load Gold Labels
# =========================
print("\U0001F4C1 Loading gold test labels...")

def load_gold_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0].isdigit():
                continue
            token, label = line.split()
            labels.append(label)
    return labels

gold_path = "/content/drive/MyDrive/545FinalMyImplementation/gua_spa_train_dev_test_gold/gua_spa_test_gold.txt"
gold_labels = load_gold_labels(gold_path)

# =========================
# 3. Prepare Feature Extraction for XGBoost
# =========================
print("\U0001F4C1 Loading Word2Vec embeddings...")
embedding_path = "/content/drive/MyDrive/545FinalMyImplementation/Word_Embedding/Vectores19.bin"
word_vec = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

print("\U0001F4C1 Loading fine-tuned mBERT (mlm_guarani)...")
bert_path = "/content/drive/MyDrive/545FinalMyImplementation/mlm_guarani"
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
bert_model = BertForTokenClassification.from_pretrained(bert_path, num_labels=3)
bert_model.eval()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# =========================
# 4. Feature Extraction Functions
# =========================
def get_word2vec_vector(token):
    try:
        return word_vec[token]
    except KeyError:
        return np.zeros(word_vec.vector_size)

def get_mbert_contextual_embedding(token):
    inputs = tokenizer(token, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model.bert(**inputs)
        vector = outputs.last_hidden_state[0, 1:-1].mean(dim=0)
    return vector.cpu().numpy()

def hybrid_embedding(token):
    w2v = get_word2vec_vector(token)
    contextual = get_mbert_contextual_embedding(token)
    return np.concatenate((w2v, contextual))

# =========================
# 5. Prepare Train and Test Features for XGBoost
# =========================
print("\U0001F4C1 Preparing features...")

# Train features
X_train, y_train = [], []
for ex in tqdm(train_dev, desc="Extracting Train Features"):
    for token, label in zip(ex["tokens"], ex["labels"]):
        if label != -100:
            X_train.append(hybrid_embedding(token))
            y_train.append(label)

# Test features
X_test, test_tokens = [], []
for ex in tqdm(test_set, desc="Extracting Test Features"):
    for token in ex["tokens"]:
        X_test.append(hybrid_embedding(token))
        test_tokens.append(token)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)

# =========================
# 6. Train XGBoost Model
# =========================
print("\U0001F4C1 Training XGBoost Hybrid model...")
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",
    device="cuda"
)
xgb_model.fit(X_train, y_train)

# =========================
# 7. Predict on Test Set
# =========================
print("\U0001F4C1 Predicting on test set...")
y_pred = xgb_model.predict(X_test)

# =========================
# 8. Align and Evaluate
# =========================
# Map label numbers to label strings if needed (depends on your mapping)
# For now assuming: 0=cc, 1=ul, 2=other or mix

print("\U0001F4C1 Aligning predictions with gold labels...")

# If gold labels and predictions match in length:
assert len(gold_labels) == len(y_pred), "Mismatch between number of test gold labels and predictions!"

# Map numeric preds to label names (assuming mapping)
label_mapping = {0: "cc", 1: "ul", 2: "other"}
y_pred_labels = [label_mapping[p] for p in y_pred]

print("\U0001F4C1 Scoring...")
print(classification_report(gold_labels, y_pred_labels, digits=4))

print("✅ Full Test Set Evaluation Complete!")
