import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load filtered data
df_merkel = pd.read_csv("../data/filtered/df_merkel.csv")
df_spd = pd.read_csv("../data/filtered/df_spd.csv")


# Add labels: Merkel=0, SPD=1
df_merkel["label"] = 0
df_spd["label"] = 1

# Combine data
df_all = pd.concat([df_merkel, df_spd], ignore_index=True)
df_all = df_all[["text", "label"]]

# Split data (70% train, 30% test)
train_df, test_df = train_test_split(
    df_all, test_size=0.3, stratify=df_all["label"], random_state=42
)

# Ensure output folder exists
os.makedirs("../data/final", exist_ok=True)

# Save to CSV
train_df.to_csv("../data/final/train.csv", index=False)
test_df.to_csv("../data/final/test.csv", index=False)

print("✅ Train/test split completed!")
print(f"Train set: {len(train_df)} entries")
print(f"Test set: {len(test_df)} entries")
