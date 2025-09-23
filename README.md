# 📊 Sentiment Analysis with Bi-LSTM

A deep learning project for sentiment classification using the
[Sentiment Analysis
Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?utm_source=chatgpt.com&select=train.csv).
The goal is to train and evaluate models that predict sentiment
(positive, negative, neutral) from text data.

**🚀 Demo App:** [Sentiment Analysis Web
App](https://sentimental-analysis-lstm-b5.streamlit.app/)

------------------------------------------------------------------------

## 📦 Dataset Description

-   **Name:** Sentiment Analysis Dataset
-   **Source:** Kaggle --- by user *abhi8923shriv*
-   **Format:** CSV files (train.csv, test.csv if available)

### Columns in `train.csv`:

-   **text** → The text content (tweets/reviews/sentences).
-   **sentiment** → Target label (positive / negative / neutral).

------------------------------------------------------------------------

## ✅ Task

-   **Objective:** Sentiment Classification
-   **Goal:** Predict the sentiment of a given text input.
-   **Labels:** Positive, Negative, Neutral

------------------------------------------------------------------------

## 🔄 Approach

### 🔹 Preprocessing

-   Text cleaning (lowercasing, punctuation removal, stopwords)
-   Tokenization & padding sequences
-   Handled noisy data (typos, emojis, special chars)

### 🔹 Models Tried

1.  **LSTM:**
    -   Initial experiments with a standard LSTM gave poor accuracy and
        unreliable predictions.
2.  **Bi-LSTM (Bidirectional LSTM):**
    -   Switching to Bi-LSTM significantly improved results.
    -   Achieved **83% accuracy** with consistent and correct
        predictions.

------------------------------------------------------------------------

## 📊 Results

-   **Final Model:** Bi-LSTM
-   **Accuracy:** 83%
-   **Predictions:** Stable and true compared to standard LSTM

------------------------------------------------------------------------

## 🔍 Usage Suggestions

**Feature Engineering Options:**
- Bag-of-Words / TF-IDF
- Word Embeddings (Word2Vec, GloVe, FastText)
- Transformer models (BERT, RoBERTa, DistilBERT)

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Macro/Weighted F1 for imbalanced data

------------------------------------------------------------------------

## ⚠ Considerations

-   Class imbalance may affect results
-   Text contains noise (typos, emojis, special chars)
-   Risk of overfitting on small dataset → apply dropout/regularization
-   Ensure proper train/test split (avoid leakage)

------------------------------------------------------------------------

## 🖼️ App Screenshots

<img width="1019" height="513" alt="image" src="https://github.com/user-attachments/assets/b6a6023c-7997-47b0-9065-7d441c4152f5" />
<img width="1191" height="597" alt="image" src="https://github.com/user-attachments/assets/03bfc4db-73bf-4e91-86a1-ae8a72d76308" />
<img width="1049" height="542" alt="image" src="https://github.com/user-attachments/assets/05dcea70-93f2-4ed9-a874-71b3fcb76a73" />

------------------------------------------------------------------------

## 📝 Licensing & Citation

-   Refer to the Kaggle dataset page for licensing terms.
-   Cite the dataset author: *abhi8923shriv* when using in
    research/publications.

------------------------------------------------------------------------

✨ With the switch from **LSTM → Bi-LSTM**, this project now achieves
**83% accuracy** and provides reliable sentiment predictions.
