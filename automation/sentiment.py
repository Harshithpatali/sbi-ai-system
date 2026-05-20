import os
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from scipy.special import softmax

from utils.logger import logger


# =========================================
# MODEL CONFIG
# =========================================

MODEL_NAME = "ProsusAI/finbert"

NEWS_PATH = (
    "data/raw/financial_news.csv"
)


# =========================================
# LOAD FINBERT
# =========================================

print("\nLoading FinBERT Model...\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

print("FinBERT Loaded Successfully\n")


# =========================================
# SENTIMENT LABELS
# =========================================

labels = [

    "negative",

    "neutral",

    "positive"
]


# =========================================
# ANALYZE SINGLE HEADLINE
# =========================================

def analyze_sentiment(text):

    try:

        inputs = tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

        scores = softmax(
            outputs.logits.numpy()[0]
        )

        sentiment_dict = {

            labels[i]: float(scores[i])

            for i in range(len(labels))
        }

        sentiment = labels[
            scores.argmax()
        ]

        return (

            sentiment,

            sentiment_dict
        )

    except Exception as e:

        logger.error(
            f"Sentiment analysis error: {e}"
        )

        return None, None


# =========================================
# AGGREGATE SENTIMENT SCORE
# =========================================

def calculate_final_sentiment(results):

    score = 0

    for item in results:

        sentiment = item["sentiment"]

        confidence = item["confidence"]

        if sentiment == "positive":

            score += confidence

        elif sentiment == "negative":

            score -= confidence

    return round(score, 2)


# =========================================
# RUN FULL SENTIMENT PIPELINE
# =========================================

def run_sentiment_pipeline():

    logger.info(
        "Starting FinBERT sentiment analysis"
    )

    # =====================================
    # CHECK FILE EXISTS
    # =====================================

    if not os.path.exists(NEWS_PATH):

        logger.warning(
            "financial_news.csv not found"
        )

        return {

            "overall_sentiment": "Neutral",

            "sentiment_score": 0,

            "total_articles": 0,

            "details": []
        }

    # =====================================
    # LOAD CSV
    # =====================================

    df = pd.read_csv(NEWS_PATH)

    # =====================================
    # EMPTY FILE CHECK
    # =====================================

    if df.empty:

        logger.warning(
            "financial_news.csv is empty"
        )

        return {

            "overall_sentiment": "Neutral",

            "sentiment_score": 0,

            "total_articles": 0,

            "details": []
        }

    # =====================================
    # VERIFY COLUMN
    # =====================================

    if "title" not in df.columns:

        logger.warning(
            "'title' column missing"
        )

        return {

            "overall_sentiment": "Neutral",

            "sentiment_score": 0,

            "total_articles": 0,

            "details": []
        }

    # =====================================
    # REMOVE EMPTY HEADLINES
    # =====================================

    df.dropna(

        subset=["title"],

        inplace=True
    )

    sentiment_results = []

    print("\nAnalyzing Headlines...\n")

    for index, row in df.iterrows():

        headline = row["title"]

        sentiment, scores = analyze_sentiment(
            headline
        )

        if sentiment is not None:

            confidence = max(
                scores.values()
            )

            result = {

                "headline": headline,

                "sentiment": sentiment,

                "confidence": round(
                    confidence * 100,
                    2
                ),

                "scores": scores
            }

            sentiment_results.append(result)

            print(
                f"\nHeadline: {headline}"
            )

            print(
                f"Sentiment: {sentiment}"
            )

            print(
                f"Confidence: "
                f"{confidence*100:.2f}%"
            )

    # =====================================
    # FINAL SENTIMENT SCORE
    # =====================================

    final_score = calculate_final_sentiment(
        sentiment_results
    )

    # =====================================
    # OVERALL MARKET SENTIMENT
    # =====================================

    if final_score > 20:

        overall_sentiment = "Bullish"

    elif final_score < -20:

        overall_sentiment = "Bearish"

    else:

        overall_sentiment = "Neutral"

    # =====================================
    # FINAL OUTPUT
    # =====================================

    final_results = {

        "overall_sentiment": overall_sentiment,

        "sentiment_score": final_score,

        "total_articles": len(
            sentiment_results
        ),

        "details": sentiment_results
    }

    logger.info(
        f"Sentiment Results: {final_results}"
    )

    return final_results


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    results = run_sentiment_pipeline()

    print("\nFinal Sentiment Analysis\n")

    print(
        f"Overall Sentiment: "
        f"{results['overall_sentiment']}"
    )

    print(
        f"Sentiment Score: "
        f"{results['sentiment_score']}"
    )

    print(
        f"Articles Analyzed: "
        f"{results['total_articles']}"
    )