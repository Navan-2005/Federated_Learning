import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "diabetes"
CATEGORICAL = ["gender", "smoking_history"]


class Preprocessor:
    """Fitted once on the training data; reused by every client and the server
    so that all models see identical feature order and scaling."""

    def __init__(self):
        self.columns = None
        self.scaler = StandardScaler()

    def _encode(self, df):
        features = df.drop(columns=[TARGET], errors="ignore")
        encoded = pd.get_dummies(features, columns=CATEGORICAL, dtype=float)
        if self.columns is not None:
            encoded = encoded.reindex(columns=self.columns, fill_value=0.0)
        return encoded

    def fit(self, df):
        self.columns = list(self._encode(df).columns)
        self.scaler.fit(self._encode(df).values)
        return self

    def transform(self, df):
        X = self.scaler.transform(self._encode(df).values).astype(np.float32)
        y = df[TARGET].values.astype(np.int64) if TARGET in df else None
        return X, y

    @property
    def num_features(self):
        return len(self.columns)
