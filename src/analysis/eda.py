from src.analysis.statistics import *
def generate_eda_report(df): return {'shape':df.shape,'summary':numeric_summary(df),'correlation':correlation_matrix(df),'class_distribution':class_distribution(df),'missing_values':df.isna().sum()}
