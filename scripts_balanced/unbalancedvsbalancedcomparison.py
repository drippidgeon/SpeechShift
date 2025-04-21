import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Paths
test_path = "../data/final/balanced/run3/test.csv"  # You can change this
balanced_model_path = "../models/bert-balanced-run3"
unbalanced_model_path = "../models/bert-merkel-spd-final"

# Load test set
df_test = pd.read_csv(test_path)
ds_test = Dataset.from_pandas(df_test)

# Tokenizer (same for both models)
tokenizer = BertTokenizer.from_pretrained(balanced_model_path)

# Tokenize test data
ds_test = ds_test.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=512), batched=True)
ds_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def evaluate_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    preds = trainer.predict(ds_test)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    return acc, f1_macro, f1_weighted

# Evaluate both models
acc_bal, f1_macro_bal, f1_weighted_bal = evaluate_model(balanced_model_path)
acc_unb, f1_macro_unb, f1_weighted_unb = evaluate_model(unbalanced_model_path)

# Print comparison
print("Evaluation Summary:")
print(f"Balanced BERT:   Accuracy={acc_bal:.4f} | Macro F1={f1_macro_bal:.4f} | Weighted F1={f1_weighted_bal:.4f}")
print(f"Unbalanced BERT: Accuracy={acc_unb:.4f} | Macro F1={f1_macro_unb:.4f} | Weighted F1={f1_weighted_unb:.4f}")

# Plot
labels = ["Balanced", "Unbalanced"]
accs = [acc_bal, acc_unb]
f1_macros = [f1_macro_bal, f1_macro_unb]
f1_weighted = [f1_weighted_bal, f1_weighted_unb]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, accs, width, label='Accuracy', color='skyblue')
rects2 = ax.bar(x, f1_macros, width, label='Macro F1', color='salmon')
rects3 = ax.bar(x + width, f1_weighted, width, label='Weighted F1', color='mediumseagreen')

ax.set_ylabel("Score")
ax.set_title("Model Comparison: Balanced vs Unbalanced")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for r in rects:
        height = r.get_height()
        ax.annotate(f'{height:.2f}', xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig("../results/Balanced_vs_UnbalancedModel.png", dpi=300)
plt.show()
