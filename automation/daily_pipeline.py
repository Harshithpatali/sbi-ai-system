from automation.fetch_stock import (
    fetch_stock_data
)

from automation.fetch_news import (
    fetch_financial_news
)

from training.features import (
    generate_features
)

from automation.database import (
    save_prediction
)

from automation.signal_engine import (
    generate_signal
)

from automation.notify import (
    create_message,
    send_telegram_alert
)

from utils.logger import logger


# =========================================
# MAIN DAILY PIPELINE
# =========================================

def run_daily_pipeline():

    print(
        "\nStarting SBI AI Daily Pipeline\n"
    )

    logger.info(
        "Daily pipeline started"
    )

    # =====================================
    # STEP 1 — FETCH STOCK DATA
    # =====================================

    print(
        "\nFetching Stock Data...\n"
    )

    fetch_stock_data()

    # =====================================
    # STEP 2 — FETCH NEWS
    # =====================================

    print(
        "\nFetching Financial News...\n"
    )

    fetch_financial_news()

    # =====================================
    # STEP 3 — FEATURE ENGINEERING
    # =====================================

    print(
        "\nGenerating Features...\n"
    )

    generate_features()

    # =====================================
    # STEP 4 — GENERATE SIGNAL
    # =====================================

    print(
        "\nGenerating AI Signal...\n"
    )

    results = generate_signal()

    # =====================================
    # STEP 5 — SAVE TO DATABASE
    # =====================================

    print(
        "\nSaving Results to Database...\n"
    )

    save_prediction(results)

    # =====================================
    # STEP 6 — SEND TELEGRAM ALERT
    # =====================================

    print(
        "\nSending Telegram Notification...\n"
    )

    message = create_message(results)

    send_telegram_alert(message)

    print(
        "\nDaily Pipeline Completed Successfully\n"
    )

    logger.info(
        "Daily pipeline completed"
    )


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    run_daily_pipeline()