
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "emails.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== SPAMGUARD AI EDA ==========\n")

# 1. Dataset information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# 2. Class distribution
print("\nEmail Classification Distribution:")
print(df["label"].value_counts())

# 3. Percentage distribution
print("\nPercentage Distribution:")
print(df["label"].value_counts(normalize=True) * 100)

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 6. Calculate email lengths
df["email_length"] = (
    df["email_subject"].fillna("") + " " +
    df["email_body"].fillna("")
).str.len()

print("\nAverage Email Length:")
print(df.groupby("label")["email_length"].mean())

# 7. Email distribution chart
plt.figure(figsize=(7, 5))

df["label"].value_counts().plot(
    kind="bar",
    color=["steelblue", "tomato"]
)

plt.title("Spam vs Ham Email Distribution")
plt.xlabel("Email Category")
plt.ylabel("Number of Emails")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "email_distribution.png")
plt.show()

# 8. Email length comparison
plt.figure(figsize=(8, 5))

for label in ["spam", "ham"]:
    df[df["label"] == label]["email_length"].plot(
        kind="hist",
        bins=40,
        alpha=0.5,
        label=label
    )

plt.title("Email Length Distribution")
plt.xlabel("Email Length (Characters)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "email_length.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", OUTPUT_DIR)
