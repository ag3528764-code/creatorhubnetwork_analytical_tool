import pandas as pd
from src.features.feature_engineering import add_engineered_features
def test_features():
 d=pd.DataFrame({'website_visits':[10],'pages_viewed':[20],'email_clicks':[4],'avg_session_minutes':[10],'previous_purchases':[4],'membership_years':[3],'days_since_last_purchase':[20]}); x=add_engineered_features(d); assert {'engagement_score','purchase_activity','recent_customer'}<=set(x.columns)
