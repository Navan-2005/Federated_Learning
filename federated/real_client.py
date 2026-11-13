"""Federated client for real deployments: trains on the participant's own CSV.

Usage (private mode: data never leaves this machine):
    python real_client.py --session-id <ID> --data C:\\path\\to\\my_data.csv --url http://SERVER:8000

The CSV must have the same columns as the sample dataset (age, gender,
hypertension, heart_disease, smoking_history, bmi, HbA1c_level,
blood_glucose_level, diabetes). The raw data never leaves this machine; only
model weights and a 2x2 validation confusion matrix are sent to the server.
"""
import argparse
import random
import time

import pandas as pd
import requests
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from async_client import predict
from client import Client
from common import state_from_json, state_to_json
from fixed_preprocessing import (
    CATEGORIES, NUM_FEATURES, NUMERIC_RANGES, TARGET, preprocess,
)
from network import ANN_Model


def run(args):
    torch.set_num_threads(1)
    session = requests.Session()
    session.headers["X-Session-Id"] = args.session_id

    if args.id is None:  # private mode: ask the server for an id and its settings
        reg = session.post(f"{args.url}/register/{args.session_id}", timeout=30)
        if reg.status_code != 200:
            raise SystemExit(f"Could not join: {reg.status_code} {reg.json().get('detail')}")
        info = reg.json()
        args.id = info["client_id"]
        args.epochs = args.epochs or info["epochs"]
        args.lr = args.lr or info["lr"]
        print(f"[client {args.id}] joined session (data stays on this machine)", flush=True)
    args.epochs = args.epochs or 5
    args.lr = args.lr or 0.01
    rng = random.Random(args.id)

    df = pd.read_csv(args.data)
    missing = [c for c in [*NUMERIC_RANGES, *CATEGORIES, TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    df = df.dropna(subset=[*NUMERIC_RANGES, *CATEGORIES, TARGET])
    stratify = df[TARGET] if df[TARGET].nunique() > 1 else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=0, stratify=stratify)
    X_train, y_train = preprocess(train_df)
    X_val, y_val = preprocess(val_df)
    print(f"[client {args.id}] {len(train_df)} train rows, {len(val_df)} validation rows", flush=True)

    client = Client(args.id, X_train, y_train)
    model = ANN_Model(input_features=NUM_FEATURES)

    pushes = 0
    while True:
        resp = session.get(f"{args.url}/model", timeout=30).json()
        if resp["done"]:
            break
        model.load_state_dict(state_from_json(resp["weights"]))
        base_version = resp["version"]

        val_cm = confusion_matrix(y_val, predict(model, X_val), labels=[0, 1]).tolist()
        weights, n, loss = client.train(model, args.epochs, args.lr)
        time.sleep(rng.uniform(0, args.max_delay))

        result = session.post(
            f"{args.url}/update",
            json={
                "client_id": args.id,
                "base_version": base_version,
                "num_samples": n,
                "weights": state_to_json(weights),
                "val_confusion_matrix": val_cm,
            },
            timeout=30,
        ).json()
        pushes += 1
        print(
            f"[client {args.id}] push {pushes} | base v{base_version} | loss {loss:.4f} | "
            f"{'accepted' if result['accepted'] else 'rejected: ' + result['reason']}",
            flush=True,
        )
        if result.get("done"):
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True, help="session id printed by the server")
    p.add_argument("--id", type=int, default=None, help="leave unset to be assigned by the server")
    p.add_argument("--data", required=True, help="path to this client's private CSV")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-delay", type=float, default=1.0)
    run(p.parse_args())
