import logging
import os
from datetime import datetime

def setup_logging(result_dir: str, log_filename: str = "training_log.txt"):
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, log_file