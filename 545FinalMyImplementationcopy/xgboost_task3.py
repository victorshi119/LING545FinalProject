# xgboost_task3.py
from datasets import load_from_disk
import numpy as np
import xgboost as xgb
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# === Load dataset ===
dataset = load_from_disk("/content/drive/MyDrive/545FinalMyImplementation/tokenized_task3_dataset")

# === Load pretrained embeddings ===
embedding_path = "/content/drive/MyDrive/545FinalMyImplementation/cc.es.300.vec"
ft_model = KeyedVectors.load_word2vec_format(embedding_path)

def get_token_vector(token):
    try:
        return ft_model[token]
    except KeyError:
        return np.zeros(ft_model.vector_size)

def extract_features_and_labels(split_data):
    X, y = [], []
    for ex in tqdm(split_data, desc="Extracting"):
        for token, label in zip(ex["tokens"], ex["labels"]):
            X.append(get_token_vector(token))
            y.append(label)
    return np.array(X), np.array(y)

# === Extract ===
X_train, y_train = extract_features_and_labels(dataset["train"])
X_dev, y_dev = extract_features_and_labels(dataset["dev"])

# === Encode labels ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_dev_enc = label_encoder.transform(y_dev)

# === Train XGBoost ===
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",  # use 'hist' + device='cuda' for GPU
    device="cuda"
)
xgb_model.fit(X_train, y_train_enc)

# === Save Model ===
joblib.dump(xgb_model, "/content/drive/MyDrive/545FinalMyImplementation/xgb_task3_model.joblib")

# === Evaluate ===
y_pred = xgb_model.predict(X_dev)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_dev_labels = label_encoder.inverse_transform(y_dev_enc)

print("=== Classification Report ===")
print(classification_report(y_dev_labels, y_pred_labels, digits=4))
print("✅ Task 3 XGBoost training and evaluation complete!")
# Save the label encoder for future use
joblib.dump(label_encoder, "/content/drive/MyDrive/545FinalMyImplementation/xgb_task3_label_encoder.joblib")
# Save the model
joblib.dump(xgb_model, "/content/drive/MyDrive/545FinalMyImplementation/xgb_task3_model.joblib")
print("✅ Model and label encoder saved.")