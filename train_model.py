
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Handle missing values
df["email_subject"] = df["email_subject"].fillna("")
df["email_body"] = df["email_body"].fillna("")

# Combine subject and body
df["email_text"] = (
    df["email_subject"] + " " + df["email_body"]
).str.strip()

# Remove empty emails
df = df[df["email_text"] != ""]

# Features and labels
X = df["email_text"]
y = df["label"]

print("Total emails:", len(df))
print("\nClass distribution:")
print(y.value_counts())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))


# Define models
models = {
    "Naive Bayes": MultinomialNB(alpha=0.5),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = {}
trained_models = {}


# Train and evaluate models
for model_name, classifier in models.items():

    print("\n" + "=" * 45)
    print("Training:", model_name)
    print("=" * 45)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                stop_words="english"
            )
        ),
        ("classifier", classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, pos_label="spam"
    )

    recall = recall_score(
        y_test, predictions, pos_label="spam"
    )

    f1 = f1_score(
        y_test, predictions, pos_label="spam"
    )

    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    results[model_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "macro_f1": float(macro_f1)
    }

    trained_models[model_name] = pipeline

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


# Select best model using macro F1
best_model_name = max(
    results,
    key=lambda name: results[name]["macro_f1"]
)

best_model = trained_models[best_model_name]

# Save selected model
model_path = MODEL_DIR / "spam_classifier.pkl"

joblib.dump(best_model, model_path)

# Save model metrics
metrics_data = {
    "results": results,
    "best_model": best_model_name,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

metrics_path = MODEL_DIR / "model_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics_data, file, indent=4)


# Generate confusion matrix for best model
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["ham", "spam"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(cmap="Blues")

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

plt.show()


# Final output
print("\n" + "=" * 45)
print("MODEL TRAINING COMPLETED")
print("=" * 45)

print("\nModel Comparison:")

for name, metrics in results.items():

    print("\n", name)

    for metric, value in metrics.items():

        if metric in ["accuracy"]:
            print(metric, ":", round(value * 100, 2), "%")
        else:
            print(metric, ":", round(value, 4))


print("\nBest Model:", best_model_name)
print("Saved model:", model_path)
print("Saved metrics:", metrics_path)
print("\nSpamGuard AI model training successful!")
