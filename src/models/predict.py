from pathlib import Path
from datetime import datetime
import joblib,pandas as pd
from src.config.config import *
from src.features.feature_engineering import add_engineered_features
from src.utils.validators import validate_customer_input
class PredictionService:
    def __init__(self,path=TRAINED_MODEL_PATH):
        if not Path(path).exists(): raise FileNotFoundError('Train the model before starting the API.')
        self.model=joblib.load(path)
    def predict(self,payload):
        data=validate_customer_input(payload); frame=add_engineered_features(pd.DataFrame([data])); p=int(self.model.predict(frame)[0]); prob=float(self.model.predict_proba(frame)[0,1]); result={'prediction':p,'prediction_label':TARGET_NAMES[p],'purchase_probability':round(prob,6),'prediction_time':datetime.now().isoformat(timespec='seconds')}; self._history({**data,**result}); return result
    def _history(self,row):
        Path(PREDICTION_HISTORY_PATH).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(PREDICTION_HISTORY_PATH,mode='a',header=not Path(PREDICTION_HISTORY_PATH).exists(),index=False)
