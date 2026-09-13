import os
import re
import pandas as pd

BRAND_LOOKUP = {
    "APPLE": "APPLE", "IPHONE": "APPLE", "IOS": "APPLE",
    "SAMSUNG": "SAMSUNG", "GALAXY": "SAMSUNG",
    "GOOGLE": "GOOGLE", "PIXEL": "GOOGLE",
    "XIAOMI": "XIAOMI", "REDMI": "XIAOMI", "POCO": "XIAOMI", "MI": "XIAOMI",
    "ONEPLUS": "ONEPLUS", "NORD": "ONEPLUS",
    "REALME": "REALME", "OPPO": "OPPO", "VIVO": "VIVO", "IQOO": "VIVO",
    "MOTOROLA": "MOTOROLA", "MOTO": "MOTOROLA",
    "NOKIA": "NOKIA", "ASUS": "ASUS", "SONY": "SONY", "LG": "LG",
    "HUAWEI": "HUAWEI", "HONOR": "HONOR", "INFINIX": "INFINIX",
    "TECNO": "TECNO", "NOTHING": "NOTHING", "LAVA": "LAVA", "MICROMAX": "MICROMAX"
}

def extract_brand_from_text(row, title_col, text_col):
    combined_text = f"{str(row.get(title_col, ''))} {str(row.get(text_col, ''))}".upper()
    for kw, target_brand in BRAND_LOOKUP.items():
        if re.search(r'\b' + kw + r'\b', combined_text):
            return target_brand
    return "UNKNOWN"

def load_and_clean_data(raw_csv_path, processed_csv_path):
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    df = df_raw.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    text_col = next((col for col in ["body", "review", "review_text", "text"] if col in df.columns), None)
    rating_col = next((col for col in ["rating", "ratings", "score"] if col in df.columns), None)
    title_col = next((col for col in ["title", "review_title", "header"] if col in df.columns), None)
    asin_col = "asin" if "asin" in df.columns else None

    if not text_col or not rating_col:
        raise ValueError("Could not dynamically locate review text or rating columns.")

    df = df.dropna(subset=[text_col, rating_col])

    # Step 1: Initial text-based brand extraction
    df["brand"] = df.apply(lambda row: extract_brand_from_text(row, title_col, text_col), axis=1)

    # Step 2: Propagate detected brand across all reviews sharing the same ASIN
    if asin_col:
        # Find the most frequent non-UNKNOWN brand for each ASIN
        known_brands = df[df["brand"] != "UNKNOWN"].groupby(asin_col)["brand"].agg(lambda x: x.mode()[0] if not x.empty else "UNKNOWN")
        
        # Map back to full dataset
        df["brand_mapped"] = df[asin_col].map(known_brands)
        df["brand"] = df["brand_mapped"].fillna(df["brand"])
        df = df.drop(columns=["brand_mapped"])

    df["clean_review"] = df[text_col].astype(str).apply(lambda s: re.sub(r"\s+", " ", s).strip())
    df = df.drop_duplicates(subset=["clean_review"])
    df["review_length"] = df["clean_review"].apply(lambda s: len(s.split()))
    df = df[df["review_length"] >= 3]

    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    df.to_csv(processed_csv_path, index=False)
    print(f"[✓] Processed dataset successfully saved ({df.shape[0]} rows) -> {processed_csv_path}")
    return df