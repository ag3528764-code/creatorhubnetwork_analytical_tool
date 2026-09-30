from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
ROOT=Path(__file__).resolve().parents[2]
DATA_DIR=ROOT/'data'; RAW_DATA_DIR=DATA_DIR/'raw'; PROCESSED_DATA_DIR=DATA_DIR/'processed'; PREDICTION_DIR=DATA_DIR/'predictions'
MODELS_DIR=ROOT/'models'; REPORTS_DIR=ROOT/'reports'; FIGURES_DIR=REPORTS_DIR/'figures'; LOGS_DIR=ROOT/'logs'
RAW_DATA_PATH=RAW_DATA_DIR/'customer_behavior_raw.csv'; CLEAN_DATA_PATH=PROCESSED_DATA_DIR/'customer_behavior_clean.csv'
TRAINED_MODEL_PATH=MODELS_DIR/'best_model.pkl'; MODEL_METADATA_PATH=MODELS_DIR/'model_metadata.json'; MODEL_RESULTS_PATH=REPORTS_DIR/'model_results.csv'
CV_RESULTS_PATH=REPORTS_DIR/'cross_validation_results.csv'; TUNING_RESULTS_PATH=REPORTS_DIR/'tuning_results.csv'; PREDICTION_HISTORY_PATH=PREDICTION_DIR/'prediction_history.csv'
RANDOM_STATE=42; TEST_SIZE=.20; N_SAMPLES=5000; TARGET_COLUMN='purchased'
NUMERICAL_FEATURES=['age','income','website_visits','previous_purchases','avg_session_minutes','pages_viewed','email_clicks','support_tickets','days_since_last_purchase','membership_years','engagement_score','purchase_activity','recent_customer']
CATEGORICAL_FEATURES=['gender','membership_type','device']; TARGET_NAMES={0:'No Purchase',1:'Purchase'}
HOST=os.getenv('HOST','127.0.0.1'); PORT=int(os.getenv('PORT','5000')); DEBUG=os.getenv('FLASK_DEBUG','true').lower()=='true'
def ensure_directories():
    for d in [RAW_DATA_DIR,PROCESSED_DATA_DIR,PREDICTION_DIR,MODELS_DIR,REPORTS_DIR,FIGURES_DIR,LOGS_DIR]: d.mkdir(parents=True,exist_ok=True)
