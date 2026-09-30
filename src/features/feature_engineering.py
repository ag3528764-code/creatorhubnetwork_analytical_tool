def add_engineered_features(df):
    x=df.copy()
    x['engagement_score']=x['website_visits']+x['pages_viewed']*.5+x['email_clicks']*1.5+x['avg_session_minutes']*.3
    x['purchase_activity']=x['previous_purchases']/(x['membership_years']+1)
    x['recent_customer']=(x['days_since_last_purchase']<=30).astype(int)
    return x
