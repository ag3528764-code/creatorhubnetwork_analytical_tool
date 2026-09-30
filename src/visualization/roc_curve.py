import joblib,matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from src.config.config import *
from src.models.train import prepare_dataset
def main():
 m=joblib.load(TRAINED_MODEL_PATH); d=prepare_dataset(); X=d[NUMERICAL_FEATURES+CATEGORICAL_FEATURES]; y=d[TARGET_COLUMN]; _,Xt,_,yt=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE); prob=m.predict_proba(Xt)[:,1]; fpr,tpr,_=roc_curve(yt,prob); auc=roc_auc_score(yt,prob); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(fpr,tpr,label=f'Selected Model (AUC={auc:.3f})'); ax.plot([0,1],[0,1],'--',label='Random Baseline'); ax.set_title('ROC Curve — Purchase Prediction'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR/'roc_curve.png',dpi=200); plt.close(fig)
if __name__=='__main__': main()
