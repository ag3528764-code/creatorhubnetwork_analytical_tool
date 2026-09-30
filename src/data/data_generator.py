import numpy as np,pandas as pd
from src.config.config import *
from src.utils.logger import logger
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))
def generate_dataset(n_samples=N_SAMPLES,random_state=RANDOM_STATE):
    r=np.random.default_rng(random_state)
    age=r.integers(18,66,n_samples); income=np.clip(r.normal(70000,30000,n_samples),18000,250000)
    visits=r.poisson(12,n_samples); purchases=r.poisson(4,n_samples); session=np.clip(r.normal(11,5,n_samples),1,60)
    pages=np.clip(visits*r.normal(2.4,.55,n_samples),1,120).round(); clicks=r.poisson(3,n_samples); tickets=r.poisson(1.5,n_samples)
    recency=np.clip(r.gamma(2,45,n_samples),0,720).round(); years=np.clip(r.normal(3.5,2.5,n_samples),0,15).round(1)
    gender=r.choice(['Male','Female','Other'],n_samples,p=[.48,.48,.04]); membership=r.choice(['Basic','Silver','Gold','Platinum'],n_samples,p=[.35,.30,.22,.13]); device=r.choice(['Mobile','Desktop','Tablet'],n_samples,p=[.55,.35,.10])
    mb=pd.Series(membership).map({'Basic':0,'Silver':.15,'Gold':.35,'Platinum':.5}).to_numpy()
    score=-.9+.000004*income+.045*visits+.13*purchases+.025*session+.015*pages+.055*clicks-.035*tickets-.004*recency+.045*years+mb+r.normal(0,.55,n_samples)
    purchased=r.binomial(1,sigmoid(score))
    df=pd.DataFrame({'age':age,'income':income.round(2),'website_visits':visits,'previous_purchases':purchases,'avg_session_minutes':session.round(2),'pages_viewed':pages.astype(int),'email_clicks':clicks,'support_tickets':tickets,'days_since_last_purchase':recency.astype(int),'membership_years':years,'gender':gender,'membership_type':membership,'device':device,'purchased':purchased})
    for c in ['income','avg_session_minutes','membership_type']:
        idx=r.choice(n_samples,max(1,int(.02*n_samples)),replace=False); df.loc[idx,c]=np.nan
    return df
def main():
    ensure_directories(); df=generate_dataset(); df.to_csv(RAW_DATA_PATH,index=False); logger.info('Generated %s',df.shape)
if __name__=='__main__': main()
