"""Federated client for real deployments: trains on the participant's own CSV.

Usage (private mode: data never leaves this machine):
    python real_client.py --session-id <ID> --data C:\\path\\to\\my_data.csv --url http://SERVER:8000

Final evaluation (no training; data still stays local, only a confusion matrix is shared):
    python real_client.py --evaluate --session-id <ID> --data C:\\path\\to\\held_out.csv --url http://SERVER:8000 --name hospital_a
The server combines all reports at GET /eval-report.

The CSV must have columns pixel_0 ... pixel_783 and label (0-9). The raw data never leaves this machine; only
model weights and a 10x10 validation confusion matrix are sent to the server.
"""
import argparse
import random
import socket
import time

import pandas as pd
import requests
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from client import Client
from common import state_from_json, state_to_json
from fixed_preprocessing import NUM_CLASSES, NUM_FEATURES, TARGET, preprocess, validate
from network import ANN_Model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).argmax(dim=1).numpy()


def evaluate_only(args):
    """Score the global model on the last 20% of local rows (no training) and share only the confusion matrix."""
    session = requests.Session()
    session.headers["X-Session-Id"] = args.session_id
    try:
        df = validate(pd.read_csv(args.data))
    except ValueError as e:
        raise SystemExit(f"Invalid CSV: {e}")
    df = df.iloc[int(len(df) * 0.8):]  # same last-20% hold-out that training never uses
    X, y = preprocess(df)

    model = ANN_Model(input_features=NUM_FEATURES, out_features=NUM_CLASSES)
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file, map_location="cpu"))
        version = args.model_version
    else:
        resp = session.get(f"{args.url}/model", timeout=60)
        if resp.status_code != 200:
            raise SystemExit(f"Could not fetch model: {resp.status_code} {resp.json().get('detail')}")
        resp = resp.json()
        model.load_state_dict(state_from_json(resp["weights"]))
        version = resp["version"]

    preds = predict(model, X)
    cm = confusion_matrix(y, preds, labels=list(range(NUM_CLASSES)))
    print(f"[{args.name}] model v{version} | {len(y)} rows | accuracy {(preds == y).mean():.4f}")
    print(cm)

    if not args.no_report:
        r = session.post(
            f"{args.url}/eval-report",
            json={"reporter": args.name, "version": version, "confusion_matrix": cm.tolist()},
            timeout=30,
        )
        if r.status_code != 200:
            raise SystemExit(f"Report rejected: {r.status_code} {r.json().get('detail')}")
        print("Confusion matrix (counts only) sent to the server; no data left this machine.")


def run(args):
    if args.evaluate:
        return evaluate_only(args)
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
    args.lr = args.lr or 0.001
    rng = random.Random(args.id)

    try:
        df = validate(pd.read_csv(args.data))
    except ValueError as e:
        raise SystemExit(f"Invalid CSV: {e}")
    cut = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]  # last 20% of the file is held out
    X_train, y_train = preprocess(train_df)
    X_val, y_val = preprocess(val_df)
    print(f"[client {args.id}] {len(train_df)} train rows, {len(val_df)} validation rows", flush=True)

    client = Client(args.id, X_train, y_train)
    model = ANN_Model(input_features=NUM_FEATURES, out_features=NUM_CLASSES)

    pushes = 0
    while True:
        try:
            resp = session.get(f"{args.url}/model", timeout=30).json()
            model.load_state_dict(state_from_json(resp["weights"]))
            base_version = resp["version"]

            val_cm = confusion_matrix(
                y_val, predict(model, X_val), labels=list(range(NUM_CLASSES))
            ).tolist()
            weights, n, loss = client.train(
                model, args.epochs, args.lr, desc=f"[client {args.id}] push {pushes + 1}"
            )
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
        except requests.ConnectionError:
            print(f"[client {args.id}] server is no longer reachable - stopping.", flush=True)
            break
        pushes += 1
        tqdm.write(
            f"[client {args.id}] push {pushes} | base v{base_version} | loss {loss:.4f} | "
            f"{'accepted' if result['accepted'] else 'rejected: ' + result['reason']}"
        )
    print(f"[client {args.id}] finished after {pushes} pushes.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True, help="session id printed by the server")
    p.add_argument("--id", type=int, default=None, help="leave unset to be assigned by the server")
    p.add_argument("--data", required=True, help="path to this client's private CSV")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-delay", type=float, default=1.0)
    p.add_argument("--evaluate", action="store_true",
                   help="evaluate-only: score the current global model on ALL local rows, no training")
    p.add_argument("--name", default=socket.gethostname(), help="reporter name used in --evaluate mode")
    p.add_argument("--model-file", default=None, help="--evaluate: use this .pt file instead of the server's model")
    p.add_argument("--model-version", type=int, default=-1, help="--evaluate with --model-file: version label to report")
    p.add_argument("--no-report", action="store_true", help="--evaluate: print locally, don't send counts to the server")
    run(p.parse_args())
