import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from client import Client
from evaluation import evaluate
from network import ANN_Model
from preprocessing import Preprocessor
from server import Server

NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 5
LR = 0.01
SEED = 20

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "dataset.csv")


def split_iid(X, y, num_clients, rng):
    idx = rng.permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, num_clients)]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["diabetes"]
    )

    prep = Preprocessor().fit(train_df)
    X_train, y_train = prep.transform(train_df)
    X_test, y_test = prep.transform(test_df)

    shards = split_iid(X_train, y_train, NUM_CLIENTS, np.random.default_rng(SEED))
    clients = [Client(i, X, y) for i, (X, y) in enumerate(shards)]

    global_model = ANN_Model(input_features=prep.num_features)
    server = Server(global_model, clients, X_test, y_test)
    server.train(ROUNDS, LOCAL_EPOCHS, LR)

    final = evaluate(global_model, X_test, y_test)
    print("\nFinal accuracy:", round(final["accuracy"], 4))
    print("Confusion matrix (rows=actual, cols=predicted):\n", final["confusion_matrix"])

    torch.save(global_model.state_dict(), os.path.join(HERE, "global_model.pt"))


if __name__ == "__main__":
    main()
