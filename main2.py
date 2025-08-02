import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

MODEL_PKL = "model.pkl"
PIPELINE_PKL = "pipeline.pkl"  # ✅ Fixed filename typo

def pipeline_builder(df_nums, df_cat):
    number_pipeline = Pipeline([
        ("scler", StandardScaler()),  # ✅ Kept your name 'scler'
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore')),
    ])
    full_pipeline = ColumnTransformer([
        ("nums", number_pipeline, df_nums.columns.to_list()),
        ("cat", cat_pipeline, df_cat.columns.to_list()),
    ])
    return full_pipeline

# ✅ Corrected logic: train if model/pipeline does NOT exist
if not os.path.exists(MODEL_PKL) or not os.path.exists(PIPELINE_PKL):
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
    df_target = df['selling_price'].copy()


    # Apply log1p transformation to target  
    df_target_log = np.log1p(df_target)

    df_features = df.drop("selling_price", axis=1)

    # Separate numerical and categorical attributes
    df_nums = df_features.select_dtypes(include=np.number)
    df_cat = df_features.select_dtypes(exclude=np.number)

    pipeline = pipeline_builder(df_nums, df_cat)
    df_pred = pipeline.fit_transform(df_features)

    # Train XGBoost model
    model = XGBRegressor()
    model.fit(df_pred, df_target_log)

    # Save model and pipeline
    joblib.dump(model, MODEL_PKL)
    joblib.dump(pipeline, PIPELINE_PKL)

else:
    modle = joblib.load(MODEL_PKL)          
    pipeline = joblib.load(PIPELINE_PKL)

    # Load new input data
    input_data = pd.read_csv("input.csv")

    # Transform input data using the pipeline
    tansfo_data = pipeline.transform(input_data)  # ✅ Typo kept as-is

    # Make predictions and reverse log1p
    pred = modle.predict(tansfo_data)
    input_data['selling_price'] = np.expm1(pred)  # ✅ Reversed log1p

    # Save output
    input_data.to_csv("Output.csv", index=False)
