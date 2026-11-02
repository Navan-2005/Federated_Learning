import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor

SEED = 20
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "dataset.csv")


def load_data(num_clients):
    """Deterministic data preparation: every process (server and each client)
    calls this with the same seed and gets identical splits and scaling."""
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["diabetes"]
    )
    prep = Preprocessor().fit(train_df)
    X_train, y_train = prep.transform(train_df)
    X_test, y_test = prep.transform(test_df)

    idx = np.random.default_rng(SEED).permutation(len(X_train))
    shards = [(X_train[s], y_train[s]) for s in np.array_split(idx, num_clients)]
    return prep, shards, (X_test, y_test)


def state_to_json(state_dict):
    return {k: v.tolist() for k, v in state_dict.items()}


def state_from_json(payload):
    return {k: torch.tensor(v) for k, v in payload.items()}
