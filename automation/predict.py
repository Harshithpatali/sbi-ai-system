import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import StandardScaler

from training.model import PatchTST

from utils.logger import logger


# =========================================
# CONFIG
# =========================================

MODEL_PATH = (
    "models/saved_models/patchtst_sbi.pth"
)

DATA_PATH = (
    "data/processed/sbi_features.csv"
)

LOOKBACK = 60


# =========================================
# FEATURES
# =========================================

FEATURE_COLUMNS = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "CLOSE_LAG_1",
    "CLOSE_LAG_2",
    "CLOSE_LAG_3",

    "VOLUME_LAG_1",

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
    "HIGH_LOW_SPREAD"
]

TARGET_COLUMN = "TARGET"


# =========================================
# DEVICE
# =========================================

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================
# LOAD MODEL
# =========================================

def load_model(input_dim):

    model = PatchTST(
        input_dim=input_dim
    )

    model.load_state_dict(

        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    logger.info(
        "Model loaded successfully"
    )

    return model


# =========================================
# PREPARE LATEST SEQUENCE
# =========================================

def prepare_latest_sequence():

    df = pd.read_csv(DATA_PATH)

    # =====================================
    # FEATURE SELECTION
    # =====================================

    features = df[FEATURE_COLUMNS]

    # =====================================
    # TARGET
    # TARGET NOW REPRESENTS:
    # NEXT-DAY % RETURN
    # =====================================

    target = df[[TARGET_COLUMN]]

    # =====================================
    # FEATURE NORMALIZATION
    # =====================================

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    # =====================================
    # TARGET NORMALIZATION
    # =====================================

    y_scaler = StandardScaler()

    y_scaler.fit(target)

    # =====================================
    # LAST 60 DAYS
    # =====================================

    latest_sequence = scaled_features[
        -LOOKBACK:
    ]

    return (

        latest_sequence,

        df,

        y_scaler
    )


# =========================================
# CONFIDENCE SCORE
# =========================================

def calculate_confidence(

    predicted_return_percent
):

    confidence = max(

        50,

        100 - abs(predicted_return_percent)
    )

    return round(confidence, 2)


# =========================================
# PREDICT NEXT DAY
# =========================================

def predict_next_day():

    logger.info(
        "Starting inference pipeline"
    )

    # =====================================
    # PREPARE DATA
    # =====================================

    latest_sequence, df, y_scaler = (
        prepare_latest_sequence()
    )

    input_dim = latest_sequence.shape[1]

    # =====================================
    # LOAD MODEL
    # =====================================

    model = load_model(input_dim)

    # =====================================
    # CONVERT TO TENSOR
    # =====================================

    input_tensor = torch.tensor(

        latest_sequence,

        dtype=torch.float32

    ).unsqueeze(0).to(DEVICE)

    # =====================================
    # INFERENCE
    # =====================================

    with torch.no_grad():

        prediction = model(input_tensor)

    # =====================================
    # CONVERT TO NUMPY
    # =====================================

    prediction_array = (

        prediction

        .cpu()

        .numpy()

        .reshape(-1, 1)
    )

    # =====================================
    # INVERSE TRANSFORM
    # PREDICTED % RETURN
    # =====================================

    predicted_return_percent = (

        y_scaler.inverse_transform(

            prediction_array

        )[0][0]
    )

    # =====================================
    # CURRENT PRICE
    # =====================================

    current_price = float(
        df["Close"].iloc[-1]
    )

    # =====================================
    # COMPUTE FUTURE PRICE
    # =====================================

    predicted_price = (

        current_price *

        (
            1 +

            predicted_return_percent / 100
        )
    )

    # =====================================
    # CONFIDENCE
    # =====================================

    confidence = calculate_confidence(

        predicted_return_percent
    )

    # =====================================
    # SIGNAL
    # =====================================

    if predicted_return_percent > 2:

        signal = "STRONG BUY"

    elif predicted_return_percent > 0:

        signal = "BUY"

    elif predicted_return_percent < -2:

        signal = "STRONG SELL"

    elif predicted_return_percent < 0:

        signal = "SELL"

    else:

        signal = "HOLD"

    # =====================================
    # RESULTS
    # =====================================

    results = {

        "current_price": round(
            current_price, 2
        ),

        "predicted_price": round(
            float(predicted_price), 2
        ),

        "predicted_return_percent": round(
            float(predicted_return_percent), 2
        ),

        "movement_percent": round(
            float(predicted_return_percent), 2
        ),

        "confidence": confidence,

        "signal": signal
    }

    logger.info(
        f"Prediction Results: {results}"
    )

    return results


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    results = predict_next_day()

    print("\nSBI Forecast Results\n")

    print(
        f"Current Price: "
        f"{results['current_price']}"
    )

    print(
        f"Predicted Return: "
        f"{results['predicted_return_percent']}%"
    )

    print(
        f"Predicted Price: "
        f"{results['predicted_price']}"
    )

    print(
        f"Signal: "
        f"{results['signal']}"
    )

    print(
        f"Confidence: "
        f"{results['confidence']}%"
    )