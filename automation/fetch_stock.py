import os
import time
import yfinance as yf
import pandas as pd

from utils.logger import logger
from config import STOCK_SYMBOL


DATA_DIR = "data/raw"


def fetch_stock_data(

    period="5y",

    interval="1d"
):

    """
    Fetch historical stock data
    with retry logic
    """

    try:

        logger.info(
            f"Fetching stock data for "
            f"{STOCK_SYMBOL}"
        )

        for attempt in range(5):

            try:

                stock = yf.download(

                    STOCK_SYMBOL,

                    period=period,

                    interval=interval,

                    auto_adjust=True,

                    progress=False,

                    threads=False
                )

                if not stock.empty:

                    break

            except Exception as e:

                logger.warning(
                    f"Attempt {attempt+1} failed: {e}"
                )

                time.sleep(5)

        if stock.empty:

            raise Exception(
                "No stock data fetched"
            )

        # =====================================
        # RESET INDEX
        # =====================================

        stock.reset_index(inplace=True)

        # =====================================
        # CREATE DIRECTORY
        # =====================================

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        file_path = os.path.join(

            DATA_DIR,

            "sbi_stock.csv"
        )

        # =====================================
        # SAVE CSV
        # =====================================

        stock.to_csv(

            file_path,

            index=False
        )

        logger.info(
            f"Stock data saved to "
            f"{file_path}"
        )

        return stock

    except Exception as e:

        logger.error(
            f"Error fetching stock data: {e}"
        )

        return None


if __name__ == "__main__":

    data = fetch_stock_data()

    if data is not None:

        print(data.head())