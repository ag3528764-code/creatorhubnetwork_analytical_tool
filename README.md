# AI-Based Data Analysis & Prediction System

**Technical title:** AI-Based Customer Behavior Analysis and Purchase Prediction System Using Machine Learning

A modular end-to-end ML application covering synthetic data generation, cleaning, EDA, feature engineering, preprocessing, model comparison, cross-validation, hyperparameter tuning, model persistence, explainability-oriented feature importance, Flask deployment, REST APIs, validation, logging, and tests.

> **Dataset note:** This repository generates a synthetic 5,000-record customer-behavior dataset. It is a reproducible project implementation, not proprietary company source code or an internal internship dataset.

## Architecture
```text
Raw/Synthetic Data
       |
       v
Validation + Cleaning
       |
       v
EDA + Statistics + Visualizations
       |
       v
Feature Engineering
       |
       v
Preprocessing Pipeline
(Imputation + Scaling + Encoding)
       |
       v
+-------------------------------+
| Logistic Regression           |
| Random Forest                 |----> Evaluation -> CV -> Tuning
| Gradient Boosting             |
+-------------------------------+                  |
                                                     v
                                            Joblib Model Artifact
                                                     |
                              +----------------------+----------------+
                              |                                       |
                              v                                       v
                        Flask Dashboard                         REST API
                              |                                       |
                              +----------------------+----------------+
                                                     v
                                        Prediction History + Logs
```

## Quick start
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.data_generator
python -m src.models.train
python -m src.models.cross_validation
python -m src.models.hyperparameter_tuning
python -m src.visualization.plots
python -m src.models.feature_importance
python -m src.visualization.roc_curve
python -m src.visualization.precision_recall
python -m src.models.evaluate
pytest
python app/app.py
```
Open `http://127.0.0.1:5000`.

## Structure
```text
app/                 Flask web application
src/                 Data, features, ML, analysis, visualization, utilities
tests/               Automated tests
data/                Raw, processed and prediction data
models/              Saved model artifacts
reports/             Metrics and figures
docs/                Architecture, API and deployment documentation
```

## Main outputs
- cleaned customer dataset
- EDA figures
- model comparison metrics
- cross-validation metrics
- tuning results
- confusion matrix
- ROC and precision-recall curves
- feature importance
- trained model
- Flask prediction interface
- REST prediction API
- prediction history
- logs
