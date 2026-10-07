# 📱 Smartphone Aspect-Based Sentiment Analysis (ABSA) Pipeline

An end-to-end NLP data engineering pipeline that analyzes **57,828 smartphone reviews** to extract fine-grained customer sentiment at the specific product feature level (camera, battery, screen, performance) across major manufacturers.

> **Context:** Developed as a modular Python package featuring automated data cleaning, dependency parsing, ASIN-level metadata propagation, and supervised machine learning classification.

---

## 📌 Project Overview

Standard star ratings mask critical product insights by assigning a single score to an entire review. Customers frequently express mixed opinions in a single statement—for example, praising the camera while criticizing battery life.

This project implements an **Aspect-Based Sentiment Analysis (ABSA)** engine that:
1. Identifies specific product feature targets (*camera*, *battery*, *display*, *price*) using syntactic dependency parsing.
2. Measures opinion polarity using rule-based sentiment scoring.
3. Groups and attributes feedback accurately to specific brands (*Samsung*, *Apple*, *Motorola*, *Google*, etc.).
4. Classifies overall document sentiment using supervised machine learning (**83.75% accuracy**).

---

## 📊 Dataset & Scope

* **Dataset Size:** 57,828 cleaned customer reviews
* **Extracted Aspect-Opinion Pairs:** 1,686 structured feature tuples
* **Brand Coverage:** Samsung, Apple, Motorola, Nokia, Sony, Xiaomi, Google, and more
* **Key Columns:** `asin`, `brand`, `rating`, `title`, `body`, `clean_review`, `review_length`

---

## 🏗️ Repository Structure

```text
smartphone-absa-project/
├── data/
│   ├── raw/                        # Raw scraped dataset files
│   └── processed/                  # Processed datasets and extracted aspect CSVs
│       ├── processed_data.csv
│       └── absa_structured_output.csv
├── notebooks/                      # Exploratory research notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_and_stats.ipynb
│   ├── 03_rule_based_absa.ipynb
│   └── 04_ml_sentiment_model.ipynb
├── src/                            # Modular Python package
│   ├── __init__.py
│   ├── data_processing.py          # Data ingestion & ASIN brand propagation
│   ├── absa_engine.py              # spaCy dependency parsing & VADER scoring
│   └── sentiment_model.py          # TF-IDF + Logistic Regression classifier
├── main.py                         # Automated end-to-end pipeline driver
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation