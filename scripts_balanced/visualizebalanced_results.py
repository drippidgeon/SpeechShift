import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

#  Parameters
run_id = 1
model_path = f"../models/bert-balanced-run{run_id}"
test_data_path = f"../data/final/balanced/run{run_id}/test.csv"
output_dir = f"../results/balanced data/run/run{run_id}"
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load test data
df_test = pd.read_csv(test_data_path)
ds_test = Dataset.from_pandas(df_test)
ds_test = ds_test.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=512), batched=True)
ds_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

#  Trainer for inference
training_args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=8)
trainer = Trainer(model=model, args=training_args)

# Predictions
pred_output = trainer.predict(ds_test)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)
probs = F.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()

#  1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Merkel", "SPD"])
fig_cm, ax_cm = plt.subplots()
disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
ax_cm.set_title(f"Confusion Matrix – Run {run_id}")
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
plt.show()

#  2. Confidence Histogram
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(probs[:, 0], bins=50, alpha=0.6, label="Confidence: Merkel", color="blue")
ax_hist.hist(probs[:, 1], bins=50, alpha=0.6, label="Confidence: SPD", color="red")
ax_hist.set_title(f"Prediction Confidence Distribution – Run {run_id}")
ax_hist.set_xlabel("Confidence")
ax_hist.set_ylabel("Frequency")
ax_hist.legend()
ax_hist.grid(True)
plt.savefig(f"{output_dir}/confidence_histogram.png", dpi=300)
plt.show()

# 3. Baseline Comparison Bar Chart
# BERT scores
bert_acc = accuracy_score(y_true, y_pred)
bert_f1_macro = f1_score(y_true, y_pred, average="macro")

# Random baseline
np.random.seed(42)
random_preds = np.random.choice([0, 1], size=len(y_true))
random_acc = accuracy_score(y_true, random_preds)
random_f1_macro = f1_score(y_true, random_preds, average="macro")

# Majority baseline
majority_class = pd.Series(y_true).mode()[0]
majority_preds = np.full_like(y_true, fill_value=majority_class)
majority_acc = accuracy_score(y_true, majority_preds)
majority_f1_macro = f1_score(y_true, majority_preds, average="macro")

# Plot Accuracy and Macro F1
labels = ["Random", "Majority", "BERT"]
acc_scores = [random_acc, majority_acc, bert_acc]
f1_macro_scores = [random_f1_macro, majority_f1_macro, bert_f1_macro]

x = np.arange(len(labels))
width = 0.35

fig_bar, ax_bar = plt.subplots()
rects1 = ax_bar.bar(x - width/2, acc_scores, width, label='Accuracy', color="skyblue")
rects2 = ax_bar.bar(x + width/2, f1_macro_scores, width, label='Macro F1 Score', color="salmon")

ax_bar.set_ylabel('Score')
ax_bar.set_title(f'Model Comparison – Run {run_id}')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylim(0, 1.05)
ax_bar.legend()

# Annotate bars
for rects in [rects1, rects2]:
    for r in rects:
        height = r.get_height()
        ax_bar.annotate(f'{height:.2f}', xy=(r.get_x() + r.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=["Merkel", "SPD"])
print("Per-Class Classification Report:")
print(report)

with open(f"{output_dir}/classification_report.txt", "w") as f:
    f.write(report)

print(f"BERT Accuracy: {bert_acc:.4f}")
print(f"BERT Macro F1: {bert_f1_macro:.4f}")
