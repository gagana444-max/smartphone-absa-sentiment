# 💻 Laptop Aspect-Based Sentiment Analysis (ABSA) Pipeline

A comprehensive data science portfolio project analyzing Flipkart laptop reviews to understand customer sentiment at both an overall and fine-grained aspect level. 

This project spans data cleaning, exploratory data analysis (EDA), non-parametric statistical hypothesis testing, and a supervised baseline machine learning sentiment model, laying the foundation for fine-grained aspect-level sentiment extraction.

> **Context:** Developed as part of a mentor-led Data Science Special Interest Group (SIG), following a structured weekly curriculum.

---

## 📌 Project Goal & Problem Statement

Standard star ratings and document-level sentiment analysis evaluate raw reviews as a single score, masking critical product insights. Laptop reviews frequently mix multiple opinions within a single sentence—for example, praising **battery life** ("lasts all day!") while criticizing the **display** ("screen is dim and washed out").

This project works toward **Aspect-Based Sentiment Analysis (ABSA)**: identifying specific product features mentioned in a review (*battery*, *display*, *build quality*, *performance*, etc.) and determining the sentiment expressed toward each one individually.

---

## 📊 Dataset Overview

* **Source:** Flipkart Laptop Customer Reviews
* **Dataset Size:** 16,991 reviews after cleaning, spanning 20 leading laptop brands
* **Key Raw Columns:** `product_name`, `rating`, `review`, `no_ratings`, `no_reviews`
* **Derived/Engineered Features:** `brand`, `popularity_tier`, `review_length`

---

## 🏗️ Repository Structure

```text
laptop-absa-project/
├── data/
│   ├── raw/                        # Raw scraped dataset files
│   └── processed/                  # Cleaned and feature-enriched CSVs
│       ├── processed_data.csv
│       └── processed_data_enriched.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb # Data schema audit & exploration
│   ├── 02_data_cleaning.ipynb      # Deduplication & text standardization
│   ├── 03_exploratory_data_analysis.ipynb # Brand & rating distributions
│   ├── 04_statistical_analysis.ipynb      # Hypothesis testing (Kruskal-Wallis, Chi-Square)
│   └── 05_model_baseline.ipynb      # Supervised sentiment classifier (TF-IDF + LogReg)
├── reports/
│   ├── data_cleaning_report.pdf
│   ├── eda_summary.pdf
│   ├── statistical_analysis_report.pdf
│   └── model_baseline_report.pdf
├── requirements.txt
└── README.md