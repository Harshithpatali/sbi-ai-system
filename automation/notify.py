import requests

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)

from automation.signal_engine import (
    generate_signal
)

from utils.logger import logger


# =========================================
# TELEGRAM API URL
# =========================================

TELEGRAM_URL = (

    f"https://api.telegram.org/bot"
    f"{TELEGRAM_BOT_TOKEN}/sendMessage"
)


# =========================================
# CREATE MESSAGE
# =========================================

def create_message(results):

    message = f"""
📈 SBI AI Prediction System

━━━━━━━━━━━━━━━━━━

💰 Current Price:
₹{results['current_price']}

🎯 Predicted Price:
₹{results['predicted_price']}

📊 Predicted Move:
{results['movement_percent']}%

🧠 Confidence:
{results['confidence']}%

📰 Market Sentiment:
{results['sentiment']}

📈 Sentiment Score:
{results['sentiment_score']}

⚡ Signal Score:
{results['signal_score']}

━━━━━━━━━━━━━━━━━━

🚨 FINAL SIGNAL:
{results['final_signal']}

━━━━━━━━━━━━━━━━━━
"""

    return message


# =========================================
# SEND TELEGRAM MESSAGE
# =========================================

def send_telegram_alert(message):

    try:

        payload = {

            "chat_id":
                TELEGRAM_CHAT_ID,

            "text":
                message
        }

        response = requests.post(

            TELEGRAM_URL,

            data=payload
        )

        if response.status_code == 200:

            print(
                "\nTelegram Alert Sent Successfully\n"
            )

            logger.info(
                "Telegram alert sent"
            )

        else:

            print(
                "\nTelegram Alert Failed\n"
            )

            print(response.text)

            logger.error(
                f"Telegram Error: {response.text}"
            )

    except Exception as e:

        logger.error(
            f"Telegram notification failed: {e}"
        )

        print(
            f"\nNotification Error: {e}\n"
        )


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    print(
        "\nGenerating AI Signal...\n"
    )

    results = generate_signal()

    message = create_message(results)

    print(message)

    print(
        "\nSending Telegram Alert...\n"
    )

    send_telegram_alert(message)