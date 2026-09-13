import os
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

VALID_ASPECTS = {
    "camera", "battery", "screen", "display", "processor",
    "performance", "sound", "speaker", "price", "value",
    "design", "quality", "build", "memory", "storage"
}

def extract_aspect_tuples(text):
    doc = nlp(str(text))
    pairs = []

    for token in doc:
        aspect, opinion = None, None

        if token.pos_ == "ADJ" and token.head.pos_ in ["NOUN", "PROPN"]:
            noun = token.head.text.lower()
            if noun in VALID_ASPECTS:
                aspect = noun
                opinion = token.text.lower()

        elif token.pos_ in ["NOUN", "PROPN"] and token.text.lower() in VALID_ASPECTS:
            for child in token.children:
                if child.dep_ in ["amod", "acomp"] and child.pos_ == "ADJ":
                    aspect = token.text.lower()
                    opinion = child.text.lower()
                    break
            if not aspect and token.head.pos_ == "ADJ" and token.dep_ in ["nsubj", "nsubjpass"]:
                aspect = token.text.lower()
                opinion = token.head.text.lower()

        if aspect and opinion:
            score = analyzer.polarity_scores(opinion)["compound"]
            pairs.append({"aspect": aspect, "opinion": opinion, "sentiment_score": score})

    return pairs

def run_absa_pipeline(df, output_csv_path, max_samples=3000):
    df_sample = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    records = []

    for _, row in df_sample.iterrows():
        pairs = extract_aspect_tuples(row["clean_review"])
        for p in pairs:
            records.append({
                "brand": row.get("brand", "UNKNOWN"),
                "aspect": p["aspect"],
                "opinion": p["opinion"],
                "sentiment_score": p["sentiment_score"]
            })

    df_absa = pd.DataFrame(records)
    
    if df_absa.empty:
        print("[!] Warning: No aspect-opinion pairs were extracted.")
        df_absa = pd.DataFrame(columns=["brand", "aspect", "opinion", "sentiment_score"])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_absa.to_csv(output_csv_path, index=False)
    print(f"[✓] ABSA extraction pipeline completed ({df_absa.shape[0]} aspect pairs) -> {output_csv_path}")
    return df_absa