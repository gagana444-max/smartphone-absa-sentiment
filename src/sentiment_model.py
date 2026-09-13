import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train_sentiment_classifier(df):
    """Trains a TF-IDF + Logistic Regression sentiment classifier."""
    rating_col = next((col for col in ["rating", "ratings", "score"] if col in df.columns), None)
    
    def label_sentiment(r):
        return "Negative" if r <= 2 else ("Neutral" if r == 3 else "Positive")

    df["sentiment_class"] = df[rating_col].apply(label_sentiment)

    X = df["clean_review"].fillna("").astype(str)
    y = df["sentiment_class"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[✓] Sentiment Classifier trained | Test Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    return model, vectorizer