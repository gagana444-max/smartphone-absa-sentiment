import os
import re
import pandas as pd

def load_and_clean_data(raw_csv_path, processed_csv_path):
    """
    Loads raw review CSV, standardizes schema, drops missing values,
    extracts brand names, cleans text, and saves processed output.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    df = df_raw.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Dynamic column resolution
    text_col = next((col for col in ["body", "review", "review_text", "text"] if col in df.columns), None)
    rating_col = next((col for col in ["rating", "ratings", "score"] if col in df.columns), None)

    if not text_col or not rating_col:
        raise ValueError("Could not dynamically locate review text or rating columns.")

    # Drop missing target rows
    df = df.dropna(subset=[text_col, rating_col])

    # Extract/Standardize Brand name
    if "brand" not in df.columns and "title" in df.columns:
        df["brand"] = df["title"].astype(str).str.split().str[0].str.upper()
    elif "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip().str.upper()

    # Clean text & word count filter
    df["clean_review"] = df[text_col].astype(str).apply(lambda s: re.sub(r"\s+", " ", s).strip())
    df = df.drop_duplicates(subset=["clean_review"])
    df["review_length"] = df["clean_review"].apply(lambda s: len(s.split()))
    df = df[df["review_length"] >= 3]

    # Save to disk
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    df.to_csv(processed_csv_path, index=False)
    print(f"[✓] Processed dataset successfully saved ({df.shape[0]} rows) -> {processed_csv_path}")
    return df