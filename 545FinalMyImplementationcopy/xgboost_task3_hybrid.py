import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# ========== 1. Load Fine-tuned mBERT Model ==========
bert_path = "/content/drive/MyDrive/545FinalMyImplementation/mlm_guarani"
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)
bert_model.eval().cuda()  # Use GPU if available

# ========== 2. Load Word2Vec Embedding ==========
embedding_path = "/content/drive/MyDrive/545FinalMyImplementation/Word_Embedding/Vectores19.bin"
embedding_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
embedding_dim = embedding_model.vector_size

# ========== 3. Load Preprocessed CSV ==========
data_path = "/content/drive/MyDrive/545FinalMyImplementation/task3_xgboost.csv"
df = pd.read_csv(data_path)
df = df.dropna(subset=["token", "label"])
df["label"] = df["label"].str.strip().str.lower()
label_map = {"cc": 0, "ul": 1}
df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# ========== 4. Feature Extraction ==========
@torch.no_grad()
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
    return cls_embedding.squeeze().cpu().numpy()

def get_word2vec_embedding(word, model, dim):
    return model[word] if word in model else np.zeros(dim)

bert_embeddings = []
word2vec_embeddings = []

for token in df["token"]:
    bert_emb = get_bert_embedding(token)  # 768-dim
    w2v_emb = get_word2vec_embedding(token, embedding_model, embedding_dim)  # 300-dim
    combined = np.concatenate([bert_emb, w2v_emb])  # (768 + 300) = 1068
    bert_embeddings.append(combined)

X = np.vstack(bert_embeddings)
y = df["label"].values

# ========== 5. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 6. Train XGBoost ==========
clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=2,
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="gpu_hist",  # Leverage T4 GPU
    predictor="gpu_predictor",
    gpu_id=0
)
clf.fit(X_train, y_train)

# ========== 7. Save Model ==========
output_model_path = "/content/drive/MyDrive/545FinalMyImplementation/xgb_task3_hybrid_model.joblib"
joblib.dump(clf, output_model_path)
print(f"✅ Model saved to {output_model_path}")

# ========== 8. Evaluate ==========
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["cc", "ul"]))


# import os
# import numpy as np
# import pandas as pd
# from transformers import BertTokenizerFast, BertModel
# from gensim.models import KeyedVectors
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import torch
# import xgboost as xgb
# import joblib

# # Paths
# bin_path = "/content/drive/MyDrive/545FinalMyImplementation/Word_Embedding/Vectores19.bin"
# model_path = "/content/drive/MyDrive/545FinalMyImplementation/mlm_guarani"
# csv_path = "/content/drive/MyDrive/545FinalMyImplementation/task3_xgboost.csv"

# # Load static word embeddings
# word_vec = KeyedVectors.load_word2vec_format(bin_path, binary=True)
# w2v_dim = word_vec.vector_size

# # Load fine-tuned mBERT and tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
# bert = BertModel.from_pretrained(model_path)
# bert.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bert.to(device)

# # Load CSV and clean
# df = pd.read_csv(csv_path)
# df = df.dropna(subset=["token", "label"])
# df["label"] = df["label"].str.strip().str.lower()
# df = df[df["label"].isin(["cc", "ul"])]
# label_map = {"cc": 0, "ul": 1}
# df["label"] = df["label"].map(label_map).astype(int)

# # Embedding fusion: contextual + static
# def hybrid_embedding(token):
#     # Word2Vec
#     w2v = word_vec[token] if token in word_vec else np.zeros(w2v_dim)

#     # mBERT contextual
#     inputs = tokenizer(token, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = bert(**inputs)
#         contextual = outputs.last_hidden_state[0, 1:-1].mean(dim=0).cpu().numpy()  # [CLS] and [SEP] removed

#     return np.concatenate((w2v, contextual))

# # Create features
# X = np.array([hybrid_embedding(tok) for tok in df["token"]])
# y = df["label"].values

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # XGBoost
# clf = xgb.XGBClassifier(
#     objective="multi:softmax",
#     num_class=2,
#     eval_metric="mlogloss"
# )
# clf.fit(X_train, y_train)

# # Save model
# output_model = "/content/drive/MyDrive/545FinalMyImplementation/xgb_task3_hybrid_model.joblib"
# joblib.dump(clf, output_model)
# print(f"✅ Model saved to {output_model}")

# # Evaluate
# y_pred = clf.predict(X_test)
# print("=== Hybrid Model Report ===")
# print(classification_report(y_test, y_pred, target_names=["cc", "ul"]))
