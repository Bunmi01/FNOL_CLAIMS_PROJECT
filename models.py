import joblib
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from huggingface_hub import hf_hub_download




# Model and feature path
Model_path = "models/best_model.pkl"
Features_path = "models/feature_columns.pkl"

Features = ['Claim_Type', 'Estimated_Claim_Amount', 'Traffic_Condition',
            'Weather_Condition', 'Vehicle_Type', 'Vehicle_Year', 
            'Driver_age_(years)', 'License_age_(years)'
]

Target = ['Ultimate_Claim_Amount']


# Functions to load model and feature columns
# def load_model():
#    with open(Model_path, 'rb') as f:
#        model = joblib.load(Model_path)
#    with open(Features_path, 'rb') as f:
#        feature_columns = joblib.load(f)
#    return model, feature_columns

REPO_ID = "Bunmi01/Ultimate_claim_cost_model"
MODEL_FILENAME = "best_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"

def load_model():
    # Download Model and feature columns from Hugging Face
    model_path = hf_hub_download(
        repo_id = REPO_ID,
        filename = MODEL_FILENAME
    )

    features_path = hf_hub_download(
        repo_id = REPO_ID,
        filename = FEATURES_FILENAME
    )

    # Load artifacts
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    return model, feature_columns


def save_model(model, versioned = False):
    if versioned:
        # find next available version number
        version = 1
        while os.path.exists(f"models/best_model_v{version}.pkl"):
            version += 1
        path = f"models/best_model_v{version}.pkl"
    else:
        path = Model_path
    with open(path, "wb") as f:
        joblib.dump(model, f)
    return path


def winsorize(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def retrain_model(new_data):

    new_data = new_data.copy() # prevent side effects

    # Derived features
    new_data["Driver_age"] = (new_data["Accident_Date"] - new_data["Date_of_Birth"]).dt.days // 365
    new_data["License_age"] = (new_data["Accident_Date"] - new_data["Full_License_issue_Date"]).dt.days // 365
    new_data["Fnol_delay"] = (new_data["FNOL_Date"] - new_data["Accident_Date"]).dt.days
    new_data["Settlement_Days"] = (new_data["Settlement_Date"] - new_data["FNOL_Date"]).dt.days

    # Fix outliers usind winsorize function
    outlier_columns =[
        "Estimated_Claim_Amount",
        "Ultimate_Claim_Amount",
        "FNOL_delay_(days)",
        "Settlement_days"
    ]

    for col in outlier_columns:
        new_data = winsorize(new_data, col)

    # Log transform target 
    new_data["Ultimate_Claim_Amount"] = np.log1p(new_data["Ultimate_Claim_Amount"])

    features = [
        'Claim_Type', 
        'Estimated_Claim_Amount', 
        'Traffic_Condition',
        'Weather_Condition', 
        'Vehicle_Type', 
        'Vehicle_Year', 
        'Driver_age_(years)', 
        'License_age_(years)'
    ]

    target = 'Ultimate_Claim_Amount'

    # Fix: warap target in list
    new_data = new_data[features + [target]]

    # One-hot encoding categorical columns
    categorical_features = [
        'Claim_Type',  
        'Traffic_Condition',
        'Weather_Condition', 
        'Vehicle_Type', 
    ]

    new_data = pd.get_dummies(
        new_data,
        columns=categorical_features,
        drop_first=False,
        dtype=int
    )

    # Load production model
    prod_model = load_model()

    # IMPORTANT FIX: align dummy columns to predict() to avoind errors
    expected_cols = list(prod_model.feature_names_in_)
    new_data = new_data.reindex(columns=expected_cols + [target])

    X = new_data.drop(columns=[target])
    y = new_data[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state= 42
    )

    # Evaluate production model
    y_pred_prod = prod_model.predict(X_test)
    rmse_prod = mean_squared_error(y_test, y_pred_prod, squared=False)

    # Train new model with same parameters
    new_model = RandomForestRegressor(**prod_model.get_params())
    new_model.fit(X_train, y_train)

    y_pred_new = new_model.predict(X_test)
    rmse_new = mean_squared_error(y_test, y_pred_new, squared=False)

    # Promote if better
    promoted = False
    if rmse_new < rmse_prod:
        save_model(new_model)
        promoted = True

    return {
        "rmse_old": rmse_prod,
        "rmse_new": rmse_new,
        "promoted": promoted
    }