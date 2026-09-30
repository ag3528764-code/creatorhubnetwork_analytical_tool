import joblib,matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from src.config.config import *
from src.models.train import prepare_dataset
def main():
    model=joblib.load(TRAINED_MODEL_PATH); df=prepare_dataset(); X=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES]; y=df[TARGET_COLUMN]; _,Xt,_,yt=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE); p=model.predict(Xt); cm=confusion_matrix(yt,p); print(classification_report(yt,p,digits=4)); print(cm)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm); fig.colorbar(im,ax=ax); ax.set_title('Confusion Matrix — Selected Model'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_xticks([0,1],['No Purchase','Purchase']); ax.set_yticks([0,1],['No Purchase','Purchase']);
    for i in range(2):
      for j in range(2): ax.text(j,i,int(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(FIGURES_DIR/'confusion_matrix_selected_model.png',dpi=200); plt.close(fig)
if __name__=='__main__': main()
