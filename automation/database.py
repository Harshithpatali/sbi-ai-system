from supabase import create_client

from config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

from automation.signal_engine import (
    generate_signal
)

from utils.logger import logger


# =========================================
# CREATE SUPABASE CLIENT
# =========================================

supabase = create_client(

    SUPABASE_URL,

    SUPABASE_KEY
)


# =========================================
# SAVE PREDICTION
# =========================================

def save_prediction(results):

    try:

        data = {

            "current_price":
                results["current_price"],

            "predicted_price":
                results["predicted_price"],

            "movement_percent":
                results["movement_percent"],

            "confidence":
                results["confidence"],

            "sentiment":
                results["sentiment"],

            "sentiment_score":
                results["sentiment_score"],

            "signal_score":
                results["signal_score"],

            "final_signal":
                results["final_signal"]
        }

        response = (

            supabase
            .table("sbi_predictions")
            .insert(data)
            .execute()
        )

        logger.info(
            "Prediction saved successfully"
        )

        return response

    except Exception as e:

        logger.error(
            f"Database insertion failed: {e}"
        )

        print(
            f"\nDatabase Error: {e}\n"
        )

        return None


# =========================================
# FETCH HISTORY
# =========================================

def fetch_prediction_history(limit=10):

    try:

        response = (

            supabase
            .table("sbi_predictions")
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .limit(limit)
            .execute()
        )

        return response.data

    except Exception as e:

        logger.error(
            f"History fetch failed: {e}"
        )

        return None


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    print(
        "\nGenerating AI Trading Signal...\n"
    )

    results = generate_signal()

    print(
        "\nSaving Results to Supabase...\n"
    )

    save_prediction(results)

    print(
        "\nFetching Recent Predictions...\n"
    )

    history = fetch_prediction_history()

    for item in history:

        print(item)