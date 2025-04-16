import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Define your topics ===
topics = {
    "Migration": ["Flüchtling", "Migration", "Asyl", "Integration"],
    "Economy & Poverty": ["Wirtschaft", "Finanzen", "Armut", "Steuern", "Haushalt"],
    "Military & War": ["Militär", "Krieg", "Verteidigung", "Bundeswehr"],
    "Climate & Renewable Energy": ["Klima", "Klimawandel", "erneuerbare Energie", "Umweltschutz"],
    "Education": ["Bildung", "Schule", "Universität", "Forschung"],
    "Pension": ["Rente", "Rentensystem", "Altersvorsorge"],
    "Inflation": ["Inflation", "Preissteigerung", "Kaufkraft"]
}

# Ensure results folder exists
results_dir = "../results/balanced data/topic_analysis"
os.makedirs(results_dir, exist_ok=True)

summary_records = []

# Evaluate each topic across all balanced models
for run_id in range(1, 6):
    print(f"\n🔄 Evaluating with model run {run_id}")

    # Load model & tokenizer
    model_path = f"../models/bert-balanced-run{run_id}"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    trainer = Trainer(model=model)

    # Load test data for the specific run
    df_test = pd.read_csv(f"../data/final/balanced/run{run_id}/test.csv")

    for topic, keywords in topics.items():
        print(f"\n🔎 Evaluating topic: {topic} (Run {run_id})")

        # Filter test data by keywords
        topic_df = df_test[df_test["text"].str.contains("|".join(keywords), case=False)]

        if topic_df.empty:
            print(f"⚠️ No speeches found for {topic}, skipping.")
            continue

        # Prepare dataset
        topic_ds = Dataset.from_pandas(topic_df)
        topic_ds = topic_ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=512), batched=True)
        topic_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Predict
        pred_output = trainer.predict(topic_ds)
        y_true = pred_output.label_ids
        y_pred = np.argmax(pred_output.predictions, axis=1)

        # Evaluate
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        print(f"✅ {topic} (Run {run_id}) - Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
        report = classification_report(y_true, y_pred, target_names=["Merkel", "SPD"])
        print(report)

        # Save results
        topic_run_path = f"{results_dir}/{topic}_run{run_id}_results.txt"
        with open(topic_run_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n")
            f.write(report)

        summary_records.append({
            "run": run_id,
            "topic": topic,
            "accuracy": acc,
            "macro_f1": macro_f1
        })

# Generate summary dataframe
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(f"{results_dir}/summary_results.csv", index=False)

# Calculate mean accuracy and macro_f1 per topic
mean_summary = summary_df.groupby('topic').agg({'accuracy': 'mean', 'macro_f1': 'mean'}).reset_index()
mean_summary.to_csv(f"{results_dir}/mean_summary_results.csv", index=False)

# Plot mean results
plt.figure(figsize=(10, 6))
x = np.arange(len(mean_summary['topic']))
width = 0.35

plt.bar(x - width/2, mean_summary['accuracy'], width, label='Mean Accuracy', color='skyblue')
plt.bar(x + width/2, mean_summary['macro_f1'], width, label='Mean Macro F1', color='salmon')

plt.ylabel('Scores')
plt.xlabel('Topics')
plt.title('Mean Accuracy and Macro F1 by Topic')
plt.xticks(x, mean_summary['topic'], rotation=45)
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig(f"{results_dir}/mean_summary_plot.png", dpi=300)
plt.show()

print("\n📄 Summary of all topics saved successfully.")
