from pathlib import Path
import pandas as pd
from flask import Flask,jsonify,render_template,request
from src.config.config import *
from src.models.predict import PredictionService
from src.utils.helpers import load_json
app=Flask(__name__,template_folder=str(Path(__file__).parent/'templates'),static_folder=str(Path(__file__).parent/'static'))
try: service=PredictionService()
except FileNotFoundError: service=None
@app.route('/')
def home(): return render_template('index.html')
@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')
@app.route('/predict')
def predict_page(): return render_template('prediction.html')
@app.get('/api/health')
def health(): return jsonify({'status':'healthy','model_loaded':service is not None})
@app.get('/api/model-info')
def info(): return jsonify(load_json(MODEL_METADATA_PATH)) if MODEL_METADATA_PATH.exists() else (jsonify({'error':'metadata unavailable'}),404)
@app.get('/api/dataset-summary')
def summary():
 if not CLEAN_DATA_PATH.exists(): return jsonify({'error':'clean dataset unavailable'}),404
 d=pd.read_csv(CLEAN_DATA_PATH); return jsonify({'rows':len(d),'columns':len(d.columns),'features':d.columns.tolist(),'missing_values':int(d.isna().sum().sum()),'duplicates':int(d.duplicated().sum()),'purchase_rate':float(d.purchased.mean())})
@app.get('/api/model-results')
def results():
 if not MODEL_RESULTS_PATH.exists(): return jsonify({'error':'model results unavailable'}),404
 return jsonify(pd.read_csv(MODEL_RESULTS_PATH).to_dict(orient='records'))
@app.get('/api/prediction-history')
def history():
 if not PREDICTION_HISTORY_PATH.exists(): return jsonify([])
 return jsonify(pd.read_csv(PREDICTION_HISTORY_PATH).tail(100).fillna('').to_dict(orient='records'))
@app.post('/api/predict')
def predict():
 if service is None: return jsonify({'error':'No trained model is loaded.'}),503
 try: return jsonify(service.predict(request.get_json(silent=True) or {}))
 except ValueError as e: return jsonify({'error':str(e)}),400
 except Exception as e: return jsonify({'error':str(e)}),500
if __name__=='__main__': app.run(host=HOST,port=PORT,debug=DEBUG)
