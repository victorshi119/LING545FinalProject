import pandas as pd

def convert_to_csv(input_path, output_path):
    rows = []
    tokens, labels = [], []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    for t, l in zip(tokens, labels):
                        rows.append({"token": t, "label": l})
                    rows.append({"token": "", "label": ""})  # Sentence break
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    tokens.append(parts[0])
                    labels.append(parts[1])

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

# Paths (modify if needed)
convert_to_csv(
    input_path="/content/drive/MyDrive/545FinalMyImplementation/gua_spa_train_dev/gua_spa_train.txt",
    output_path="/content/drive/MyDrive/545FinalMyImplementation/task3_xgboost.csv"
)
