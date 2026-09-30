from pathlib import Path
import pandas as pd
from src.config.config import RAW_DATA_PATH,CLEAN_DATA_PATH
def load_raw_data(path=RAW_DATA_PATH):
    if not Path(path).exists(): raise FileNotFoundError(f'Raw data not found: {path}')
    return pd.read_csv(path)
def load_clean_data(path=CLEAN_DATA_PATH):
    if not Path(path).exists(): raise FileNotFoundError(f'Clean data not found: {path}')
    return pd.read_csv(path)
