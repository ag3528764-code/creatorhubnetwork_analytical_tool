from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.config.config import NUMERICAL_FEATURES,CATEGORICAL_FEATURES
def build_preprocessor():
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('num',num,NUMERICAL_FEATURES),('cat',cat,CATEGORICAL_FEATURES)],remainder='drop',verbose_feature_names_out=False)
