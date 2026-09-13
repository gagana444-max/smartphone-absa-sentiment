# 📱 Smartphone Aspect-Based Sentiment Analysis (ABSA) Pipeline

An end-to-end, production-ready Natural Language Processing (NLP) pipeline that extracts granular product aspect sentiments (e.g., *battery*, *camera*, *screen*, *price*) from unstructured smartphone customer reviews.

---

## 📌 Business Problem & Impact

Standard sentiment analysis models evaluate raw reviews as a single overall score, masking critical product insights. A customer might love a phone's **camera** ("amazing photos!") while despising its **battery** ("terrible life"). 

This project solves that limitation by leveraging **Syntactic Dependency Parsing** and **Supervised Machine Learning** to isolate specific device feature mentions and accurately score aspect-level sentiments, giving product teams actionable feature feedback.

---

## 🏗️ Architecture & Pipeline Overview

The project is structured into three execution stages:

1. **Data Preprocessing & Cleaning:** Normalizes schema, extracts smartphone brands, removes noise, and filters short reviews.
2. **Rule-Based Aspect Extraction:** Employs `spaCy` dependency parsing to map descriptive adjectives directly to aspect targets, scored via `vaderSentiment`.
3. **Supervised Sentiment Modeling:** Converts review text into TF-IDF n-gram vectors and trains a `LogisticRegression` classifier for multi-class sentiment categorization.