import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np
import os
import torch


# Load datasets
train_df = pd.read_csv("../data/final/unbalanced data/train.csv")
test_df = pd.read_csv("../data/final/unbalanced data/test.csv")

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

# Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_ds = train_ds.map(tokenize, batched=True, batch_size=32)
test_ds = test_ds.map(tokenize, batched=True, batch_size=32)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Model initialization
model = BertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/bert-merkel-spd",
    eval_strategy="epoch",
    save_strategy="epoch",  #Data is big, so epoch saving incase training breaks
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Evaluation metrics
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save final model
os.makedirs("../models/bert-merkel-spd-final", exist_ok=True)
model.save_pretrained("../models/bert-merkel-spd-final")
tokenizer.save_pretrained("../models/bert-merkel-spd-final")

print("Training completed and model saved.")
