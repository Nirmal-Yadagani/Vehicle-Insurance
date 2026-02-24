import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime


# constants for logging
LOG_DIR = from_root('logs')
LOG_FILE_NAME = f'{datetime.now().strftime("%Y-%m-%d")}.log'
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 3

# Define the log directory path and log file path
LOG_DIR_PATH = os.path.join(from_root(), LOG_DIR)
os.makedirs(LOG_DIR_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, LOG_FILE_NAME)

def configure_logger():
    """Configures the logger to write logs to a file and console with rotation."""
    # create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a rotating file handler
    file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define a log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Configure the logger when the module is imported
configure_logger()
