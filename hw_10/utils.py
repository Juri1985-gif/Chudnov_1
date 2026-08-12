import os
import pickle

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_dataset():
    df = pd.read_csv("data/realty_data.csv")
    df = df[["price", "total_square", "rooms", "floor", "city", "source"]]
    df = df.dropna()
    df["rooms"] = df[["rooms"]].astype(int)
    df["floor"] = df[["floor"]].astype(int)
    return df


def train_model(train):
    X, y = train.drop(columns=["price"]), train['price']

    categorical_features = ['city', 'source']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("lgbm", LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X, y)

    with open('rf_fitted.pkl', 'wb') as file:
        pickle.dump(pipeline, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model

