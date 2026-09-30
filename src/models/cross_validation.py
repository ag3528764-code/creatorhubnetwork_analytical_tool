import pandas as pd
from sklearn.model_selection import StratifiedKFold,cross_validate
from src.config.config import *
from src.models.train import prepare_dataset,build_models,build_pipeline
def main():
    df=prepare_dataset(); X=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES]; y=df[TARGET_COLUMN]; cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE)
    scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}; rows=[]
    for n,m in build_models().items():
        r=cross_validate(build_pipeline(m),X,y,cv=cv,scoring=scoring,n_jobs=-1); rows.append({'Model':n,**{k+'_mean':r['test_'+k].mean() for k in scoring},'f1_std':r['test_f1'].std()})
    out=pd.DataFrame(rows); out.to_csv(CV_RESULTS_PATH,index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
