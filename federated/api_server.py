import hmac
import io
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pandas as pd

import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from aggregator import async_merge
from common import HERE, NUM_FEATURES, SEED, state_from_json, state_to_json
from fixed_preprocessing import CATEGORIES, NUMERIC_RANGES, TARGET
from network import ANN_Model

SESSION_ID = os.environ.get("FL_SESSION_ID", "")
SERVER_URL = os.environ.get("FL_SERVER_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_BYTES = int(os.environ.get("FL_MAX_UPLOAD_MB", 50)) * 1024 * 1024
LOCAL_EPOCHS = os.environ.get("FL_LOCAL_EPOCHS", "5")
LR = os.environ.get("FL_LR", "0.01")
REQUIRED_COLUMNS = [*NUMERIC_RANGES, *CATEGORIES, TARGET]
NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", 5))
MAX_UPDATES = int(os.environ.get("FL_MAX_UPDATES", 100))
BASE_ALPHA = float(os.environ.get("FL_BASE_ALPHA", 0.5))
MAX_STALENESS = int(os.environ.get("FL_MAX_STALENESS", 20))
EVAL_EVERY = int(os.environ.get("FL_EVAL_EVERY", 5))

app = FastAPI(title="Async Federated Learning Server")
lock = threading.Lock()

# The server holds no data: the global model starts from random weights and
# is evaluated only through confusion matrices that clients compute locally.
torch.manual_seed(SEED)
model = ANN_Model(input_features=NUM_FEATURES)

state = {"version": 0, "accepted": 0, "rejected": 0, "history": [], "client_cms": {}}


class Update(BaseModel):
    client_id: int
    base_version: int
    num_samples: int
    weights: dict
    val_confusion_matrix: list | None = None  # global model scored on the client's local validation data


def _log_metrics():
    if not state["client_cms"]:
        return
    cm = np.sum(list(state["client_cms"].values()), axis=0)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    entry = {
        "update": state["accepted"],
        "version": state["version"],
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": cm.tolist(),
    }
    state["history"].append(entry)
    print(
        f"Update {entry['update']:03d} | v{entry['version']} | acc {entry['accuracy']:.4f} "
        f"| recall {entry['recall']:.4f} | f1 {entry['f1']:.4f}",
        flush=True,
    )


def _done():
    return state["accepted"] >= MAX_UPDATES


@app.get("/model")
def get_model(x_session_id: str = Header(default="")):
    _check_session(x_session_id)
    with lock:
        return {
            "version": state["version"],
            "done": _done(),
            "weights": state_to_json(model.state_dict()),
        }


@app.post("/update")
def post_update(update: Update, x_session_id: str = Header(default="")):
    _check_session(x_session_id)
    with lock:
        if _done():
            return {"accepted": False, "reason": "training complete", "done": True}

        staleness = state["version"] - update.base_version
        if staleness < 0:
            raise HTTPException(400, "base_version is ahead of the server")
        if staleness > MAX_STALENESS:
            state["rejected"] += 1
            return {"accepted": False, "reason": "too stale", "done": False,
                    "version": state["version"]}

        if update.val_confusion_matrix is not None:
            state["client_cms"][update.client_id] = np.array(update.val_confusion_matrix)

        merged, alpha = async_merge(
            model.state_dict(), state_from_json(update.weights), BASE_ALPHA, staleness
        )
        model.load_state_dict(merged)
        state["version"] += 1
        state["accepted"] += 1

        if state["accepted"] % EVAL_EVERY == 0 or _done():
            _log_metrics()
        if _done():
            torch.save(model.state_dict(), os.path.join(HERE, "global_model_async.pt"))

        return {"accepted": True, "alpha": alpha, "staleness": staleness,
                "version": state["version"], "done": _done()}


joined = {"next_id": 0, "clients": {}}


def _check_session(session_id):
    if not SESSION_ID or not hmac.compare_digest(session_id or "", SESSION_ID):
        raise HTTPException(404, "unknown session id")


@app.post("/register/{session_id}")
def register(session_id: str):
    """Private mode: join without uploading data; the client trains on its own machine."""
    _check_session(session_id)
    with lock:
        if _done():
            raise HTTPException(409, "training already complete")
        client_id = joined["next_id"]
        joined["next_id"] += 1
        joined["clients"][client_id] = "remote"
    return {"client_id": client_id, "epochs": int(LOCAL_EPOCHS), "lr": float(LR)}


def _run_client(client_id, path):
    try:
        subprocess.run(
            [sys.executable, "real_client.py", "--id", str(client_id), "--data", path,
             "--url", SERVER_URL, "--session-id", SESSION_ID,
             "--epochs", LOCAL_EPOCHS, "--lr", LR],
            cwd=HERE,
        )
    finally:
        os.remove(path)  # the uploaded CSV is deleted as soon as the client stops
        joined["clients"][client_id] = "finished"


@app.post("/join/{session_id}")
async def join(session_id: str, file: UploadFile = File(...)):
    """Upload mode: the server receives the CSV and starts a local training worker for it."""
    _check_session(session_id)
    with lock:
        if _done():
            raise HTTPException(409, "training already complete")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "could not parse the file as CSV")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"missing columns: {missing}")
    if len(df) < 50:
        raise HTTPException(400, "need at least 50 rows")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise HTTPException(400, f"{TARGET} must contain only 0/1")
    for col, values in CATEGORIES.items():
        bad = sorted(set(df[col].dropna().unique()) - set(values))
        if bad:
            raise HTTPException(400, f"unexpected values in '{col}': {bad}")
    df = df[REQUIRED_COLUMNS].dropna()

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="fl_client_")
    os.close(fd)
    df.to_csv(path, index=False)
    with lock:
        client_id = joined["next_id"]
        joined["next_id"] += 1
        joined["clients"][client_id] = "training"
    threading.Thread(target=_run_client, args=(client_id, path), daemon=True).start()
    return {"joined": True, "client_id": client_id, "rows": len(df),
            "status_url": f"{SERVER_URL}/status"}


@app.get("/status")
def status():
    with lock:
        return {
            "version": state["version"],
            "accepted": state["accepted"],
            "rejected": state["rejected"],
            "max_updates": MAX_UPDATES,
            "done": _done(),
            "clients": dict(joined["clients"]),
            "latest": state["history"][-1] if state["history"] else None,
        }
