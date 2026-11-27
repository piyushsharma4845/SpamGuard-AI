
import joblib
import numpy as np

from pathlib import Path
from lime.lime_text import LimeTextExplainer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"


# Load trained model
model = joblib.load(MODEL_PATH)


# LIME explainer
explainer = LimeTextExplainer(
    class_names=["ham", "spam"]
)


def explain_email(email_text):
    """
    Explain why the model classified an email
    as spam or ham.
    """

    # Get prediction probabilities
    probabilities = model.predict_proba([email_text])[0]

    classes = list(model.classes_)

    # Get prediction
    prediction = model.predict([email_text])[0]

    # Generate explanation
    explanation = explainer.explain_instance(
        email_text,
        model.predict_proba,
        num_features=10,
        num_samples=1000,
        labels=[classes.index("spam")]
    )

    # Extract important words
    word_contributions = explanation.as_list(
        label=classes.index("spam")
    )

    return {
        "prediction": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": {
            class_name: float(probabilities[i])
            for i, class_name in enumerate(classes)
        },
        "important_words": word_contributions
    }


# Test the explanation system
if __name__ == "__main__":

    sample_email = """
    Congratulations! You have won a free iPhone.
    Click here to claim your prize immediately.
    """

    result = explain_email(sample_email)

    print("\n========== SPAMGUARD AI ==========")

    print("\nPrediction:", result["prediction"])

    print(
        "Confidence:",
        round(result["confidence"] * 100, 2),
        "%"
    )

    print("\nClass probabilities:")

    for label, probability in result["probabilities"].items():
        print(label, ":", round(probability * 100, 2), "%")

    print("\nImportant words influencing SPAM prediction:")

    for word, contribution in result["important_words"]:
        print(word, ":", round(contribution, 4))
