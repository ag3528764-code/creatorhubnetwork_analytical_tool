import matplotlib.pyplot as plt,seaborn as sns
from src.config.config import FIGURES_DIR,TARGET_COLUMN
from src.models.train import prepare_dataset
def main():
    FIGURES_DIR.mkdir(parents=True,exist_ok=True); df=prepare_dataset()
    plots=[('target_distribution.png',lambda ax:sns.countplot(data=df,x=TARGET_COLUMN,ax=ax),'Target Distribution — Purchase Outcome'),('income_distribution.png',lambda ax:sns.histplot(data=df,x='income',kde=True,ax=ax),'Exploratory Data Analysis — Income Distribution'),('website_visits_vs_purchase.png',lambda ax:sns.boxplot(data=df,x=TARGET_COLUMN,y='website_visits',ax=ax),'Website Visits by Purchase Outcome')]
    for fn,draw,title in plots:
        fig,ax=plt.subplots(figsize=(8,5)); draw(ax); ax.set_title(title); fig.tight_layout(); fig.savefig(FIGURES_DIR/fn,dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9)); sns.heatmap(df.select_dtypes('number').corr(),cmap='coolwarm',center=0,ax=ax); ax.set_title('Correlation Matrix'); fig.tight_layout(); fig.savefig(FIGURES_DIR/'correlation_matrix.png',dpi=200); plt.close(fig)
if __name__=='__main__': main()
