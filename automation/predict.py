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

    predicted_price,

    current_price
):

    difference = abs(
        predicted_price - current_price
    )

    percentage_change = (
        difference / current_price
    ) * 100

    confidence = max(
        50,
        100 - percentage_change
    )

    return round(confidence, 2)


# =========================================
# PREDICT NEXT DAY PRICE
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
    # =====================================

    predicted_price = y_scaler.inverse_transform(

        prediction_array

    )[0][0]

    # =====================================
    # CURRENT PRICE
    # =====================================

    current_price = float(
        df["Close"].iloc[-1]
    )

    # =====================================
    # PREDICTED MOVEMENT
    # =====================================

    movement_percent = (

        (
            predicted_price -
            current_price
        )

        / current_price

    ) * 100

    # =====================================
    # CONFIDENCE
    # =====================================

    confidence = calculate_confidence(

        predicted_price,

        current_price
    )

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

        "movement_percent": round(
            float(movement_percent), 2
        ),

        "confidence": confidence
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
        f"Predicted Price: "
        f"{results['predicted_price']}"
    )

    print(
        f"Predicted Move: "
        f"{results['movement_percent']}%"
    )

    print(
        f"Confidence: "
        f"{results['confidence']}%"
    )