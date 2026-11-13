"""Data-independent preprocessing for real deployments.

Every participant applies the same hard-coded schema and scaling ranges, so no
client needs to see another client's data (or share statistics) to agree on
feature order and scale. All clients and the server must use this same file.
"""
import numpy as np
import pandas as pd

TARGET = "diabetes"

# (min, max) domain ranges used for min-max scaling.
NUMERIC_RANGES = {
    "age": (0.0, 100.0),
    "hypertension": (0.0, 1.0),
    "heart_disease": (0.0, 1.0),
    "bmi": (10.0, 100.0),
    "HbA1c_level": (3.0, 10.0),
    "blood_glucose_level": (50.0, 350.0),
}
CATEGORIES = {
    "gender": ["Female", "Male", "Other"],
    "smoking_history": ["No Info", "current", "ever", "former", "never", "not current"],
}

FEATURE_NAMES = list(NUMERIC_RANGES) + [
    f"{col}_{v}" for col, values in CATEGORIES.items() for v in values
]
NUM_FEATURES = len(FEATURE_NAMES)  # 15


def preprocess(df):
    cols = []
    for name, (lo, hi) in NUMERIC_RANGES.items():
        cols.append(((df[name].astype(float) - lo) / (hi - lo)).clip(0, 1).to_numpy())
    for col, values in CATEGORIES.items():
        for v in values:
            cols.append((df[col] == v).astype(float).to_numpy())
    X = np.stack(cols, axis=1).astype(np.float32)
    y = df[TARGET].to_numpy().astype(np.int64)
    return X, y
