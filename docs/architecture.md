# Architecture

The system follows a layered ML architecture: data generation/loading -> cleaning -> EDA -> feature engineering -> preprocessing -> model training -> evaluation/CV/tuning -> persistence -> Flask/API -> prediction history/logging.

The `Pipeline` and `ColumnTransformer` ensure training and inference apply the same preprocessing steps.
