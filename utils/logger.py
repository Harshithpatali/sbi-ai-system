import logging
import os


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


logger = logging.getLogger("sbi_ai_system")

logger.setLevel(logging.INFO)


formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)


file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "app.log")
)

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)