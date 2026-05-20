import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


# =========================================
# CONFIG
# =========================================

DATA_PATH = "data/processed/sbi_features.csv"

LOOKBACK = 60

BATCH_SIZE = 32


# =========================================
# FEATURES USED FOR TRAINING
# =========================================

FEATURE_COLUMNS = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

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
# CUSTOM DATASET
# =========================================

class StockDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (

            torch.tensor(
                self.X[idx],
                dtype=torch.float32
            ),

            torch.tensor(
                self.y[idx],
                dtype=torch.float32
            )
        )


# =========================================
# CREATE SEQUENCES
# =========================================

def create_sequences(features, targets, lookback):

    X = []
    y = []

    for i in range(len(features) - lookback):

        X.append(
            features[i:i + lookback]
        )

        y.append(
            targets[i + lookback]
        )

    return np.array(X), np.array(y)


# =========================================
# LOAD DATASET
# =========================================

def load_data():

    print("\nLoading Processed Dataset...\n")

    df = pd.read_csv(DATA_PATH)

    # =====================================
    # SELECT FEATURES
    # =====================================

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    # =====================================
    # NORMALIZATION
    # =====================================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # =====================================
    # CREATE SEQUENCES
    # =====================================

    X_seq, y_seq = create_sequences(
        X_scaled,
        y.values,
        LOOKBACK
    )

    print(f"Sequence Shape: {X_seq.shape}")

    print(f"Target Shape: {y_seq.shape}")

    # =====================================
    # TRAIN TEST SPLIT
    # =====================================

    X_train, X_test, y_train, y_test = train_test_split(

        X_seq,
        y_seq,

        test_size=0.2,

        shuffle=False
    )

    # =====================================
    # CREATE DATASETS
    # =====================================

    train_dataset = StockDataset(
        X_train,
        y_train
    )

    test_dataset = StockDataset(
        X_test,
        y_test
    )

    # =====================================
    # DATALOADERS
    # =====================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    print("\nDataset Preparation Complete\n")

    return (

        train_loader,
        test_loader,

        scaler,

        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================
# MAIN TEST
# =========================================

if __name__ == "__main__":

    (
        train_loader,
        test_loader,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print("Train Samples:", len(X_train))

    print("Test Samples:", len(X_test))

    # Inspect one batch
    for batch_X, batch_y in train_loader:

        print("\nBatch Feature Shape:")

        print(batch_X.shape)

        print("\nBatch Target Shape:")

        print(batch_y.shape)

        break