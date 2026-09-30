# ML Pipeline

Numerical variables use median imputation and StandardScaler. Categorical variables use most-frequent imputation and OneHotEncoder. Baselines are Logistic Regression, Random Forest and Gradient Boosting. The baseline selection metric is F1-score. Five-fold StratifiedKFold and RandomizedSearchCV are included.
