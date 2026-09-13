import os
from src.data_processing import load_and_clean_data
from src.absa_engine import run_absa_pipeline
from src.sentiment_model import train_sentiment_classifier

def main():
    print("=" * 60)
    print("STARTING SMARTPHONE ABSA END-TO-END PIPELINE")
    print("=" * 60)

    # Path Configuration
    RAW_PATH = os.path.join("data", "raw", "smartphones_raw.csv")
    PROCESSED_PATH = os.path.join("data", "processed", "processed_data.csv")
    ABSA_OUTPUT_PATH = os.path.join("data", "processed", "absa_structured_output.csv")

    # Step 1: Data Cleaning & Schema Standardization
    print("\n[Step 1/3] Data Cleaning & Schema Standardization...")
    df_cleaned = load_and_clean_data(RAW_PATH, PROCESSED_PATH)

    # Step 2: Rule-Based ABSA Extraction
    print("\n[Step 2/3] Extracting Aspect-Opinion Tuples via spaCy Dependency Parsing...")
    run_absa_pipeline(df_cleaned, ABSA_OUTPUT_PATH, max_samples=3000)

    # Step 3: Train Sentiment Classifier
    print("\n[Step 3/3] Training Supervised ML Sentiment Classifier...")
    train_sentiment_classifier(df_cleaned)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()