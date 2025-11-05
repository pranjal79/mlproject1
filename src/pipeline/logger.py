import logging
import os
from datetime import datetime

# ✅ Create a logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ Create a unique log file each run
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# ✅ Configure logging for both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# ✅ Export a named logger object
logger = logging.getLogger(__name__)
