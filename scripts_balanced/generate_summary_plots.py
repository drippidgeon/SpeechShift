import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load CSV files
df_models = pd.read_csv("../results/balanced data/balanced_runs_summary.csv")
df_topics = pd.read_csv("../results/balanced data/topic_analysis/summary_results.csv")

# Output directory
output_dir = "../results/balanced data/"

# 1. Plot mean accuracy and macro F1 of balanced model runs
mean_accuracy = df_models["accuracy"].mean()
mean_macro_f1 = df_models["f1_macro"].mean()

plt.figure(figsize=(6, 5))
plt.bar(["Accuracy", "Macro F1"], [mean_accuracy, mean_macro_f1], color=["skyblue", "salmon"])
plt.ylim(0.85, 1.0)
plt.title("Mean Performance of Balanced BERT Models")
for i, value in enumerate([mean_accuracy, mean_macro_f1]):
    plt.text(i, value + 0.005, f"{value:.3f}", ha='center', fontsize=11)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_balanced_model_performance.png"), dpi=300)
plt.close()

# 2. Plot average macro F1 per topic  (The other Graph was A: hard to find, B: Not looking good)
df_topic_avg = df_topics.groupby("topic").agg({"macro_f1": "mean"}).sort_values("macro_f1", ascending=True)

plt.figure(figsize=(8, 6))
df_topic_avg["macro_f1"].plot(kind="barh", color="mediumseagreen")
plt.xlabel("Average Macro F1 Score")
plt.title("Topic-wise Classification Performance (Avg. over 5 runs)")
for index, value in enumerate(df_topic_avg["macro_f1"]):
    plt.text(value + 0.005, index, f"{value:.2f}", va='center')
plt.xlim(0.6, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "topic_macro_f1_avg.png"), dpi=300)
plt.close()

print("Figures saved to:", output_dir)
