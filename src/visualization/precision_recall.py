import joblib,matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.model_selection import train_test_split
from src.config.config import *
from src.models.train import prepare_dataset
def main():
 m=joblib.load(TRAINED_MODEL_PATH); d=prepare_dataset(); X=d[NUMERICAL_FEATURES+CATEGORICAL_FEATURES]; y=d[TARGET_COLUMN]; _,Xt,_,yt=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE); prob=m.predict_proba(Xt)[:,1]; precision,recall,_=precision_recall_curve(yt,prob); ap=average_precision_score(yt,prob); fig,ax=plt.subplots(figsize=(8,6)); ax.plot(recall,precision,label=f'Selected Model (AP={ap:.3f})'); ax.set_title('Precision-Recall Curve — Purchase Prediction'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR/'precision_recall_curve.png',dpi=200); plt.close(fig)
if __name__=='__main__': main()
