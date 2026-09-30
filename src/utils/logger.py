import logging
from src.config.config import LOGS_DIR,ensure_directories
ensure_directories(); LOG_FILE=LOGS_DIR/'application.log'
logger=logging.getLogger('ai_prediction_system'); logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    fh=logging.FileHandler(LOG_FILE,encoding='utf-8'); fh.setFormatter(fmt)
    sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(sh)
