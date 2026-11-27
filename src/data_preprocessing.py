
import tarfile
import pandas as pd
import re
from pathlib import Path
from email import policy
from email.parser import BytesParser
from io import BytesIO


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Dataset archives and their labels
DATASETS = {
    "20030228_easy_ham.tar.bz2": "ham",
    "20030228_hard_ham.tar.bz2": "ham",
    "20030228_spam.tar.bz2": "spam",
    "20050311_spam_2.tar.bz2": "spam"
}


def clean_text(text):
    """Remove unnecessary whitespace from email text."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_email_content(raw_email):
    """Extract subject and body from an email message."""

    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)

        subject = str(msg.get("Subject", ""))

        body_parts = []

        # Prefer plain-text email content
        for part in msg.walk():

            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        body_parts.append(content)
                except Exception:
                    continue

        body = " ".join(body_parts)

        # Fallback to HTML if plain text is unavailable
        if not body:
            for part in msg.walk():

                if part.get_content_type() == "text/html":
                    try:
                        body = part.get_content()
                        body = re.sub(r"<[^>]+>", " ", body)
                        break
                    except Exception:
                        continue

        subject = clean_text(subject)
        body = clean_text(body)

        return subject, body

    except Exception:
        return "", ""


def process_dataset():

    all_emails = []

    print("\nStarting dataset processing...\n")

    for archive_name, label in DATASETS.items():

        archive_path = RAW_DIR / archive_name

        if not archive_path.exists():
            print(f"Missing dataset: {archive_name}")
            continue

        print(f"Processing: {archive_name}")

        try:
            with tarfile.open(archive_path, mode="r:bz2") as archive:

                for member in archive.getmembers():

                    # Only process regular files
                    if not member.isfile():
                        continue

                    filename = Path(member.name).name

                    # Skip README and other documentation files
                    if filename.lower().startswith("readme"):
                        continue

                    file_object = archive.extractfile(member)

                    if file_object is None:
                        continue

                    raw_email = file_object.read()

                    subject, body = extract_email_content(raw_email)

                    # Skip empty messages
                    if not subject and not body:
                        continue

                    all_emails.append({
                        "email_subject": subject,
                        "email_body": body,
                        "label": label
                    })

            print(f"Completed: {archive_name}")

        except Exception as error:
            print(f"Error processing {archive_name}: {error}")

    if not all_emails:
        print("\nNo emails were extracted. Check your dataset files.")
        return

    # Convert emails into a DataFrame
    df = pd.DataFrame(all_emails)

    # Remove duplicate emails
    df.drop_duplicates(
        subset=["email_subject", "email_body"],
        inplace=True
    )

    # Remove empty emails
    df = df[
        (df["email_subject"] != "") |
        (df["email_body"] != "")
    ]

    # Save processed dataset
    output_path = PROCESSED_DIR / "emails.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n------------------------------")
    print("DATASET PROCESSING COMPLETED")
    print("------------------------------")

    print(f"Total emails: {len(df)}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    print(f"\nSaved dataset: {output_path}")


if __name__ == "__main__":
    process_dataset()
