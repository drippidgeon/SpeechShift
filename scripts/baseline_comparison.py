import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load test data
df = pd.read_csv("../data/final/unbalanced/test.csv")
y_true = df["label"]

# Check label balance
print("Label distribution in test set:")
print(df["label"].value_counts(normalize=True))

# Random baseline predictions
np.random.seed(42)
random_preds = np.random.choice([0, 1], size=len(y_true))


print("Random Baseline:")
print("Accuracy:", accuracy_score(y_true, random_preds))
print("F1 Score:", f1_score(y_true, random_preds, average="weighted"))

# Majority baseline predictions (predict most frequent class)
majority_class = y_true.mode()[0]
majority_preds = np.full(len(y_true), majority_class)

print("\n Majority Baseline:")
print("Accuracy:", accuracy_score(y_true, majority_preds))
print("F1 Score:", f1_score(y_true, majority_preds, average="weighted"))


