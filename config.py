import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# STOCK CONFIG
# =========================

STOCK_SYMBOL = "SBIN.NS"

# =========================
# DATA CONFIG
# =========================

LOOKBACK_DAYS = 60
PREDICTION_DAYS = 1

# =========================
# MODEL CONFIG
# =========================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# =========================
# API KEYS
# =========================

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")