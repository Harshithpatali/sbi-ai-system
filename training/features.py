import os
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands

from utils.logger import logger


# =========================================
# PATH CONFIGURATION
# =========================================

RAW_DATA_PATH = "data/raw/sbi_stock.csv"

PROCESSED_DATA_PATH = (
    "data/processed/sbi_features.csv"
)


# =========================================
# FEATURE ENGINEERING FUNCTION
# =========================================

def generate_features():

    try:

        logger.info(
            "Starting feature engineering pipeline"
        )

        # =========================================
        # VERIFY RAW FILE EXISTS
        # =========================================

        if not os.path.exists(
            RAW_DATA_PATH
        ):

            raise FileNotFoundError(
                f"{RAW_DATA_PATH} not found"
            )

        # =========================================
        # LOAD RAW STOCK DATA
        # =========================================

        df = pd.read_csv(
            RAW_DATA_PATH
        )

        print(
            "\nRaw Data Loaded Successfully\n"
        )

        # =========================================
        # CLEAN COLUMN NAMES
        # =========================================

        df.columns = (
            df.columns.str.strip()
        )

        # =========================================
        # VERIFY REQUIRED COLUMNS
        # =========================================

        required_columns = [

            "Date",

            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [

            col

            for col in required_columns

            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(

                f"Missing columns: "
                f"{missing_columns}"
            )

        # =========================================
        # CONVERT NUMERIC COLUMNS
        # =========================================

        numeric_columns = [

            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"
            )

        # =========================================
        # REMOVE INVALID ROWS
        # =========================================

        df.dropna(

            subset=numeric_columns,

            inplace=True
        )

        # =========================================
        # DATE CONVERSION
        # =========================================

        df["Date"] = pd.to_datetime(

            df["Date"],

            errors="coerce"
        )

        df.dropna(
            subset=["Date"],
            inplace=True
        )

        # =========================================
        # SORT DATA
        # =========================================

        df.sort_values(

            "Date",

            inplace=True
        )

        df.reset_index(

            drop=True,

            inplace=True
        )

        print(
            "\nData Cleaning Completed\n"
        )

        print(df.dtypes)

        # =========================================
        # RSI
        # =========================================

        rsi = RSIIndicator(

            close=df["Close"],

            window=14
        )

        df["RSI"] = rsi.rsi()

        # =========================================
        # EMA
        # =========================================

        ema = EMAIndicator(

            close=df["Close"],

            window=20
        )

        df["EMA_20"] = (
            ema.ema_indicator()
        )

        # =========================================
        # SMA
        # =========================================

        sma = SMAIndicator(

            close=df["Close"],

            window=20
        )

        df["SMA_20"] = (
            sma.sma_indicator()
        )

        # =========================================
        # MACD
        # =========================================

        macd = MACD(
            close=df["Close"]
        )

        df["MACD"] = (
            macd.macd()
        )

        df["MACD_SIGNAL"] = (
            macd.macd_signal()
        )

        df["MACD_HIST"] = (
            macd.macd_diff()
        )

        # =========================================
        # BOLLINGER BANDS
        # =========================================

        bb = BollingerBands(

            close=df["Close"],

            window=20,

            window_dev=2
        )

        df["BB_UPPER"] = (
            bb.bollinger_hband()
        )

        df["BB_MIDDLE"] = (
            bb.bollinger_mavg()
        )

        df["BB_LOWER"] = (
            bb.bollinger_lband()
        )

        # =========================================
        # DAILY RETURNS
        # =========================================

        df["RETURNS"] = (
            df["Close"].pct_change()
        )

        # =========================================
        # VOLATILITY
        # =========================================

        df["VOLATILITY"] = (

            df["RETURNS"]

            .rolling(window=20)

            .std()
        )

        # =========================================
        # VOLUME MOVING AVERAGE
        # =========================================

        df["VOLUME_MA"] = (

            df["Volume"]

            .rolling(window=20)

            .mean()
        )

        # =========================================
        # PRICE CHANGE
        # =========================================

        df["PRICE_CHANGE"] = (

            df["Close"] -
            df["Open"]
        )

        # =========================================
        # LAG FEATURES
        # =========================================

        df["CLOSE_LAG_1"] = (
            df["Close"].shift(1)
        )

        df["CLOSE_LAG_2"] = (
            df["Close"].shift(2)
        )

        df["CLOSE_LAG_3"] = (
            df["Close"].shift(3)
        )

        df["VOLUME_LAG_1"] = (
            df["Volume"].shift(1)
        )

        # =========================================
        # HIGH LOW SPREAD
        # =========================================

        df["HIGH_LOW_SPREAD"] = (

            df["High"] -
            df["Low"]
        )

        # =========================================
        # TARGET VARIABLE
        # NEXT DAY RETURN %
        # =========================================

        df["TARGET"] = (

            (
                df["Close"].shift(-1)
                - df["Close"]
            )

            / df["Close"]

        ) * 100

        # =========================================
        # REMOVE NaN VALUES
        # =========================================

        df.dropna(inplace=True)

        # =========================================
        # CREATE OUTPUT DIRECTORY
        # =========================================

        os.makedirs(

            "data/processed",

            exist_ok=True
        )

        # =========================================
        # SAVE FEATURED DATASET
        # =========================================

        df.to_csv(

            PROCESSED_DATA_PATH,

            index=False
        )

        logger.info(

            f"Processed dataset saved to "
            f"{PROCESSED_DATA_PATH}"
        )

        # =========================================
        # DISPLAY INFO
        # =========================================

        print(
            "\nFeature Engineering Complete\n"
        )

        print("Dataset Shape:")

        print(df.shape)

        print("\nGenerated Features:\n")

        feature_columns = [

            "RSI",

            "EMA_20",
            "SMA_20",

            "MACD",
            "MACD_SIGNAL",
            "MACD_HIST",

            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",

            "RETURNS",
            "VOLATILITY",

            "VOLUME_MA",

            "PRICE_CHANGE",

            "CLOSE_LAG_1",
            "CLOSE_LAG_2",
            "CLOSE_LAG_3",

            "VOLUME_LAG_1",

            "HIGH_LOW_SPREAD",

            "TARGET"
        ]

        for feature in feature_columns:

            print(feature)

        print("\nSample Data:\n")

        print(df.head())

        return df

    except Exception as e:

        logger.error(

            f"Feature engineering failed: {e}"
        )

        print(
            f"\nError occurred: {e}\n"
        )

        return None


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    generate_features()