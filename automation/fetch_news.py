import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime
from config import NEWS_API_KEY
from utils.logger import logger


DATA_DIR = "data/raw"


def fetch_financial_news():
    """
    Fetch SBI and banking-related news
    """

    try:
        logger.info("Fetching financial news")

        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        query = (
            "SBI OR State Bank of India OR RBI "
            "OR Indian banking OR banking sector"
        )

        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=50
        )

        news_data = []

        for article in articles["articles"]:

            news_data.append({
                "date": article["publishedAt"],
                "title": article["title"],
                "description": article["description"],
                "source": article["source"]["name"],
                "url": article["url"]
            })

        df = pd.DataFrame(news_data)

        os.makedirs(DATA_DIR, exist_ok=True)

        file_path = os.path.join(DATA_DIR, "financial_news.csv")

        df.to_csv(file_path, index=False)

        logger.info(f"News data saved to {file_path}")

        return df

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return None


if __name__ == "__main__":

    news = fetch_financial_news()

    if news is not None:
        print(f"Total Articles Fetched: {len(news)}")
        print(news.head())