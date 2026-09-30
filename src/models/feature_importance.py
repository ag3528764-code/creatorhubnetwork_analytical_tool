import joblib,numpy as np,pandas as pd,matplotlib.pyplot as plt
from src.config.config import *
def main():
    pipe=joblib.load(TRAINED_MODEL_PATH); model=pipe.named_steps['model']; pre=pipe.named_steps['preprocessor']
    if not hasattr(model,'feature_importances_'): print('Selected model has no tree feature_importances_. Run tuning to use Random Forest.'); return
    enc=pre.named_transformers_['cat'].named_steps['onehot']; names=np.r_[NUMERICAL_FEATURES,enc.get_feature_names_out(CATEGORICAL_FEATURES)]; imp=model.feature_importances_; out=pd.DataFrame({'Feature':names,'Importance':imp}).sort_values('Importance',ascending=False); out.to_csv(FIGURES_DIR/'feature_importance.csv',index=False); top=out.head(15).sort_values('Importance')
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(top.Feature,top.Importance); ax.set_title('Feature Importance — Selected Tree Model'); ax.set_xlabel('Importance'); fig.tight_layout(); fig.savefig(FIGURES_DIR/'feature_importance.png',dpi=200); plt.close(fig); print(out.head(15).to_string(index=False))
if __name__=='__main__': main()
