import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from utils.logger import logger
from config import STOCK_SYMBOL


DATA_DIR = "data/raw"


def fetch_stock_data(period="5y", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance
    """

    try:
        logger.info(f"Fetching stock data for {STOCK_SYMBOL}")

        stock = yf.download(
            STOCK_SYMBOL,
            period=period,
            interval=interval,
            auto_adjust=True
        )

        if stock.empty:
            raise Exception("No stock data fetched")

        # Reset index
        stock.reset_index(inplace=True)

        # Create directory if not exists
        os.makedirs(DATA_DIR, exist_ok=True)

        file_path = os.path.join(DATA_DIR, "sbi_stock.csv")

        stock.to_csv(file_path, index=False)

        logger.info(f"Stock data saved to {file_path}")

        return stock

    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return None


if __name__ == "__main__":
    data = fetch_stock_data()

    if data is not None:
        print(data.head())