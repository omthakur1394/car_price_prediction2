import pandas as pd
import joblib 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("data.csv")

# Drop the 'model' column if not needed
df = df.drop('model', axis=1)

# Create a price category for stratified splitting
df['price_cat'] = pd.qcut(df['selling_price'], q=5, labels=False, duplicates='drop')

# Perform stratified shuffle split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['price_cat']):
    stect_train_set = df.loc[train_index].drop('price_cat', axis=1)
    stect_test_set = df.loc[test_index].drop('price_cat', axis=1)

# Use the training set for processing
df = stect_train_set.copy()

# Separate target and features
df_target = df['selling_price'].copy()

# 👉 Apply log1p transformation to target (makes model stable)
df_target_log = np.log1p(df_target)  # ✅ LOG TRANSFORMATION

df_features = df.drop("selling_price", axis=1)

# Separate numerical and categorical attributes
df_nums = df_features.select_dtypes(include=np.number)
df_cat = df_features.select_dtypes(exclude=np.number)

# Create pipeline for numerical and categorical
number_pipeline = Pipeline([
    ("scler", StandardScaler()),
])
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore')),
])
full_pipeline = ColumnTransformer([
    ("nums", number_pipeline, df_nums.columns.to_list()),
    ("cat", cat_pipeline, df_cat.columns.to_list()),
])

# Transform features using the full pipeline
df_pred = full_pipeline.fit_transform(df_features)

# Train XGBoost model
model = XGBRegressor()
model.fit(df_pred, df_target_log)  # ✅ Train on log-transformed target

# Predict on training data and reverse the log using expm1
predicted_log = model.predict(df_pred)
predicted_original = np.expm1(predicted_log)  # ✅ Reverse the log1p

# Evaluate using cross-validation on log-transformed target
model_res = -cross_val_score(model, df_pred, df_target_log, scoring="neg_mean_squared_error", cv=10)
print("Cross-validation RMSE (log scale):")
# print(np.sqrt(model_res))
print(pd.Series(model_res).describe())
r2_scores = cross_val_score(model, df_pred, df_target_log, scoring="r2", cv=10)
print("\nCross-validation R2 Score:")
print(pd.Series(r2_scores).describe())
