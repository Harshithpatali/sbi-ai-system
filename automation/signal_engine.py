from automation.predict import predict_next_day
from automation.sentiment import run_sentiment_pipeline

from utils.logger import logger


# =========================================
# SIGNAL GENERATION
# =========================================

def generate_signal():

    logger.info(
        "Starting signal engine"
    )

    # =====================================
    # GET PREDICTION RESULTS
    # =====================================

    prediction_results = predict_next_day()

    current_price = (
        prediction_results["current_price"]
    )

    predicted_price = (
        prediction_results["predicted_price"]
    )

    movement_percent = (
        prediction_results["movement_percent"]
    )

    confidence = (
        prediction_results["confidence"]
    )

    # =====================================
    # GET SENTIMENT RESULTS
    # =====================================

    sentiment_results = (
        run_sentiment_pipeline()
    )

    overall_sentiment = (
        sentiment_results[
            "overall_sentiment"
        ]
    )

    sentiment_score = (
        sentiment_results[
            "sentiment_score"
        ]
    )

    # =====================================
    # WEIGHTED DECISION ENGINE
    # =====================================

    signal_score = 0

    # =====================================
    # PRICE PREDICTION WEIGHT
    # =====================================

    if movement_percent > 2:

        signal_score += 3

    elif movement_percent > 0.5:

        signal_score += 2

    elif movement_percent < -2:

        signal_score -= 3

    elif movement_percent < -0.5:

        signal_score -= 2

    # =====================================
    # SENTIMENT WEIGHT
    # =====================================

    if overall_sentiment == "Bullish":

        signal_score += 2

    elif overall_sentiment == "Bearish":

        signal_score -= 2

    # =====================================
    # CONFIDENCE WEIGHT
    # =====================================

    if confidence > 90:

        signal_score += 1

    elif confidence < 60:

        signal_score -= 1

    # =====================================
    # FINAL SIGNAL
    # =====================================

    if signal_score >= 5:

        final_signal = "STRONG BUY"

    elif signal_score >= 3:

        final_signal = "BUY"

    elif signal_score <= -5:

        final_signal = "STRONG SELL"

    elif signal_score <= -3:

        final_signal = "SELL"

    else:

        final_signal = "HOLD"

    # =====================================
    # FINAL RESULTS
    # =====================================

    results = {

        "current_price": current_price,

        "predicted_price": predicted_price,

        "movement_percent": movement_percent,

        "confidence": confidence,

        "sentiment": overall_sentiment,

        "sentiment_score": sentiment_score,

        "signal_score": signal_score,

        "final_signal": final_signal
    }

    logger.info(
        f"Signal Engine Results: {results}"
    )

    return results


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    results = generate_signal()

    print("\nSBI AI Trading Signal\n")

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

    print(
        f"Market Sentiment: "
        f"{results['sentiment']}"
    )

    print(
        f"Sentiment Score: "
        f"{results['sentiment_score']}"
    )

    print(
        f"Signal Score: "
        f"{results['signal_score']}"
    )

    print(
        f"\nFINAL SIGNAL: "
        f"{results['final_signal']}"
    )