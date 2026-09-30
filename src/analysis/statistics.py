import numpy as np
def numeric_summary(df): return df.describe(include='all').transpose()
def correlation_matrix(df): return df.select_dtypes(include=np.number).corr()
def class_distribution(df,target='purchased'): return df[target].value_counts().sort_index()
