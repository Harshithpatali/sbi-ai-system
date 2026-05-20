import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from training.dataset import load_data
from training.model import PatchTST

from utils.logger import logger


# =========================================
# DEVICE CONFIGURATION
# =========================================

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}\n")


# =========================================
# TRAINING CONFIG
# =========================================

EPOCHS = 20

LEARNING_RATE = 0.001

MODEL_SAVE_PATH = (
    "models/saved_models/patchtst_sbi.pth"
)


# =========================================
# MAPE METRIC
# =========================================

def mean_absolute_percentage_error(
    y_true,
    y_pred
):

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    return np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100


# =========================================
# TRAIN FUNCTION
# =========================================

def train_model():

    logger.info(
        "Starting PatchTST training"
    )

    # =====================================
    # LOAD DATA
    # =====================================

    (
        train_loader,
        test_loader,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    # =====================================
    # MODEL
    # =====================================

    input_dim = X_train.shape[2]

    model = PatchTST(
        input_dim=input_dim
    ).to(DEVICE)

    # =====================================
    # LOSS FUNCTION
    # =====================================

    criterion = nn.MSELoss()

    # =====================================
    # OPTIMIZER
    # =====================================

    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    # =====================================
    # TRAINING LOOP
    # =====================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)

            batch_y = batch_y.to(DEVICE)

            # =============================
            # ZERO GRADIENTS
            # =============================

            optimizer.zero_grad()

            # =============================
            # FORWARD PASS
            # =============================

            predictions = model(batch_X)

            # =============================
            # LOSS
            # =============================

            loss = criterion(
                predictions,
                batch_y
            )

            # =============================
            # BACKPROPAGATION
            # =============================

            loss.backward()

            # =============================
            # UPDATE WEIGHTS
            # =============================

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        print(

            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

    logger.info(
        "Training completed"
    )

    # =====================================
    # EVALUATION
    # =====================================

    model.eval()

    predictions_list = []

    actuals_list = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(DEVICE)

            outputs = model(batch_X)

            predictions_list.extend(
                outputs.cpu().numpy()
            )

            actuals_list.extend(
                batch_y.numpy()
            )

    # =====================================
    # METRICS
    # =====================================

    mae = mean_absolute_error(

        actuals_list,

        predictions_list
    )

    rmse = np.sqrt(
        mean_squared_error(
            actuals_list,
            predictions_list
        )
    )

    mape = mean_absolute_percentage_error(

        actuals_list,

        predictions_list
    )

    print("\nEvaluation Metrics\n")

    print(f"MAE  : {mae:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"MAPE : {mape:.2f}%")

    logger.info(
        f"MAE={mae}, RMSE={rmse}, MAPE={mape}"
    )

    # =====================================
    # SAVE MODEL
    # =====================================

    os.makedirs(
        "models/saved_models",
        exist_ok=True
    )

    torch.save(

        model.state_dict(),

        MODEL_SAVE_PATH
    )

    print(

        f"\nModel saved to:\n"
        f"{MODEL_SAVE_PATH}"
    )

    logger.info(
        "Model saved successfully"
    )

    return model


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    train_model()