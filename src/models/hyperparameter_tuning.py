import joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from src.config.config import *
from src.models.train import prepare_dataset
from src.data.data_preprocessing import build_preprocessor
from src.utils.helpers import save_json
def main():
    df=prepare_dataset(); X=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES]; y=df[TARGET_COLUMN]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    pipe=Pipeline([('preprocessor',build_preprocessor()),('model',RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1))])
    params={'model__n_estimators':[150,200,300,400],'model__max_depth':[None,8,12,16,20],'model__min_samples_split':[2,4,6,10],'model__min_samples_leaf':[1,2,4],'model__max_features':['sqrt','log2',None]}
    search=RandomizedSearchCV(pipe,params,n_iter=20,scoring='f1',cv=5,random_state=RANDOM_STATE,n_jobs=-1,verbose=1); search.fit(Xtr,ytr)
    pd.DataFrame(search.cv_results_)[['rank_test_score','mean_test_score','std_test_score','params']].sort_values('rank_test_score').to_csv(TUNING_RESULTS_PATH,index=False)
    joblib.dump(search.best_estimator_,TRAINED_MODEL_PATH); save_json({'selected_model':'Tuned Random Forest','selection_metric':'F1','best_cv_score':float(search.best_score_),'best_parameters':search.best_params_},MODEL_METADATA_PATH); print(search.best_params_,search.best_score_)
if __name__=='__main__': main()
