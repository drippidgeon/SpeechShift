import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch

# Load filtered speeches
df_merkel = pd.read_csv("../data/filtered/df_merkel.csv")
df_spd = pd.read_csv("../data/filtered/df_spd.csv")

# Label data
df_merkel["label"] = 0
df_spd["label"] = 1

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

# Store evaluation results
all_results = []

for run in range(1, 6):
    print(f"\n🔁 Starting Run {run}/5")

    # === 1. Balance data with different random sample ===
    df_spd_sampled = df_spd.sample(n=len(df_merkel), random_state=run)
    df_all = pd.concat([df_merkel, df_spd_sampled], ignore_index=True)
    df_all = df_all[["text", "label"]]

    # === 2. Train/test split ===
    train_df, test_df = train_test_split(
        df_all, test_size=0.3, stratify=df_all["label"], random_state=run
    )
    # Save the train and test set for this run
    os.makedirs(f"../data/final/balanced/run{run}", exist_ok=True)
    train_df.to_csv(f"../data/final/balanced/run{run}/train.csv", index=False)
    test_df.to_csv(f"../data/final/balanced/run{run}/test.csv", index=False)

    # Convert to HuggingFace datasets
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # === 3. Load model ===
    model = BertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels=2)

    # === 4. Training arguments ===
    output_dir = f"../models/bert-balanced-run{run}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",  #here I chose no since the data is way smaller so i didn't need to make sure to save inbetween
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=False,
    )

    # === 5. Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    # === 6. Train & Save  model ===
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # === 7. Evaluate ===
    pred_output = trainer.predict(test_ds)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    all_results.append({
        "run": run,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    })

    print(f"✅ Run {run} complete: Accuracy={acc:.4f}, F1_macro={f1_macro:.4f}")

# Save results to CSV
results_df = pd.DataFrame(all_results)
os.makedirs("../results/balanced data", exist_ok=True)
results_df.to_csv("../results/balanced data/balanced_runs_summary.csv", index=False)
print("\n📄 All runs complete! Summary:")
print(results_df.describe())
