import pandas as pd
from src.config.config import *
from src.data.data_loader import load_raw_data
from src.utils.logger import logger
def clean_data(df):
    x=df.copy().drop_duplicates().reset_index(drop=True)
    for c in x.select_dtypes(include='number').columns:
        if c!=TARGET_COLUMN: x[c]=x[c].fillna(x[c].median())
    for c in x.select_dtypes(include='object').columns:
        m=x[c].mode(dropna=True); x[c]=x[c].fillna(m.iloc[0] if len(m) else 'Unknown')
    return x
def main():
    ensure_directories(); x=clean_data(load_raw_data()); x.to_csv(CLEAN_DATA_PATH,index=False); logger.info('Saved clean data to %s',CLEAN_DATA_PATH)
if __name__=='__main__': main()
