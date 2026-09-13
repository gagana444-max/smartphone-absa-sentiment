"""
Data Cleaning Pipeline for Smartphone ABSA Dataset.
Loads raw smartphone review data, applies text standardization and filtering,
and exports the cleaned data to data/processed/processed_data.csv.
"""

import os
import re
import sys
import pandas as pd


def get_default_paths():
    """Resolve raw and processed dataset paths relative to current or repo directory."""
    candidates_raw = [
        os.path.abspath(os.path.join("data", "raw", "smartphones_raw.csv")),
        os.path.abspath(os.path.join("..", "data", "raw", "smartphones_raw.csv")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "smartphones_raw.csv")),
    ]

    raw_path = None
    for path in candidates_raw:
        if os.path.exists(path):
            raw_path = path
            break

    if raw_path is None:
        raise FileNotFoundError(
            "Raw dataset 'smartphones_raw.csv' not found. "
            "Please ensure it is placed in 'data/raw/smartphones_raw.csv'."
        )

    # Output path alongside data directory
    # raw_path is <project_root>/data/raw/smartphones_raw.csv
    data_dir = os.path.dirname(os.path.dirname(raw_path)) # <project_root>/data
    project_root = os.path.dirname(data_dir)             # <project_root>
    processed_path = os.path.join(project_root, "data", "processed", "processed_data.csv")

    return raw_path, processed_path


def clean_dataset(raw_path: str = None, processed_path: str = None) -> pd.DataFrame:
    """
    Clean the raw smartphone dataset and save to processed_data.csv.

    Steps performed:
    1. Load raw CSV data
    2. Standardize column names (lowercase with underscores)
    3. Identify text and rating columns dynamically
    4. Drop rows with missing text or rating
    5. Standardize/extract brand names
    6. Clean review text (normalize whitespace)
    7. Remove duplicate reviews
    8. Compute review_length and remove reviews shorter than 3 words
    9. Export cleaned dataset to CSV
    """
    if raw_path is None or processed_path is None:
        default_raw, default_processed = get_default_paths()
        raw_path = raw_path or default_raw
        processed_path = processed_path or default_processed

    print(f"[1/5] Loading raw dataset from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    initial_shape = df_raw.shape
    print(f"      Initial shape: {initial_shape}")

    df = df_raw.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Dynamically identify text and rating columns
    text_col = next((col for col in ["body", "review", "review_text", "text"] if col in df.columns), None)
    rating_col = next((col for col in ["rating", "ratings", "score"] if col in df.columns), None)

    if not text_col or not rating_col:
        raise ValueError(
            f"Could not identify required columns in dataset. Found columns: {list(df.columns)}. "
            f"Expected text column (body/review/text) and rating column (rating/ratings/score)."
        )

    print(f"[2/5] Identified columns -> text: '{text_col}', rating: '{rating_col}'")

    # 1. Drop rows missing essential review text or ratings
    df = df.dropna(subset=[text_col, rating_col])

    # 2. Extract/Standardize Brand name
    if "brand" not in df.columns and "title" in df.columns:
        df["brand"] = df["title"].astype(str).str.split().str[0].str.upper()
    elif "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.strip().str.upper()

    # 3. Clean review text (remove excess spaces)
    print("[3/5] Cleaning review text and normalizing whitespace...")
    df["clean_review"] = df[text_col].astype(str).apply(lambda s: re.sub(r"\s+", " ", s).strip())

    # 4. Remove duplicate reviews
    df = df.drop_duplicates(subset=["clean_review"])

    # 5. Add review word count feature & remove empty/short reviews
    print("[4/5] Filtering short reviews (< 3 words)...")
    df["review_length"] = df["clean_review"].apply(lambda s: len(s.split()))
    df = df[df["review_length"] >= 3]

    final_shape = df.shape
    print(f"[5/5] Cleaned dataset shape: {final_shape} (removed {initial_shape[0] - final_shape[0]} rows)")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # Export cleaned data to processed folder
    df.to_csv(processed_path, index=False)
    print(f"Successfully saved clean dataset to: {processed_path}")

    return df


def main():
    try:
        clean_dataset()
    except Exception as e:
        print(f"Error during data cleaning: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
