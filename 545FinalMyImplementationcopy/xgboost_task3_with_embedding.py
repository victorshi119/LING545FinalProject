
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# Load the word embedding model
embedding_path = "/content/drive/MyDrive/545FinalMyImplementation/Word Embedding/Vectores19.bin"
embedding_model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
embedding_dim = embedding_model.vector_size

# Load your preprocessed data
# Assume you have a CSV with columns: "tokens" (space-separated) and "label" (cc or ul)
data_path = "/content/drive/MyDrive/545FinalMyImplementation/task3_dataset/task3_xgboost.csv"
df = pd.read_csv(data_path)
df["tokens"] = df["tokens"].apply(lambda x: x.split())

label_map = {"cc": 0, "ul": 1}
df["label"] = df["label"].map(label_map)

# Function to get sentence embedding
def sentence_embedding(tokens, model, dim):
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# Extract features
X = np.array([sentence_embedding(tokens, embedding_model, embedding_dim) for tokens in df["tokens"]])
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=2,
    eval_metric="mlogloss",
    use_label_encoder=False
)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "xgb_task3_model.joblib")
print("✅ Model saved to xgb_task3_model.joblib")

# Evaluate
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["cc", "ul"]))
