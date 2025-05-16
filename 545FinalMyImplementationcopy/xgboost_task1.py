
from transformers import BertTokenizer, BertModel
from datasets import load_from_disk
import torch
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# === CONFIG ===
DATA_PATH = "/content/drive/MyDrive/545FinalMyImplementation/tokenized_task1_dataset"
MODEL_SAVE_PATH = "xgb_task1_model.joblib"
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# === Load Data ===
dataset = load_from_disk(DATA_PATH)

# === Load mBERT Tokenizer and Model ===
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased").eval().to(DEVICE)

# === Label Mapping ===
label_list = ["es", "gn", "other", "ne-b-per", "ne-i-per", "ne-b-org", "ne-i-org", "ne-b-loc", "ne-i-loc"]
label_to_id = {label: i for i, label in enumerate(label_list)}

# === Feature Extraction Function ===
def extract_features_and_labels(split):
    X, y = [], []
    for item in tqdm(dataset[split], desc=f"Extracting {split}"):
        input_ids = torch.tensor([item["input_ids"]]).to(DEVICE)
        attention_mask = torch.tensor([item["attention_mask"]]).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[0]

        for i, label in enumerate(item["labels"]):
            if label != -100:
                X.append(embeddings[i].cpu().numpy())
                y.append(label)
    return np.array(X), np.array(y)

# === Train ===
X_train, y_train = extract_features_and_labels("train")

clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=1, tree_method="gpu_hist" if USE_GPU else "auto")
clf.fit(X_train, y_train)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# === Evaluate ===
X_dev, y_dev = extract_features_and_labels("dev")
y_pred = clf.predict(X_dev)
print("=== Classification Report ===")
print(classification_report(y_dev, y_pred, target_names=label_list))
