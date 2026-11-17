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
from common import HERE, SEED, state_from_json, state_to_json
from fixed_preprocessing import NUM_CLASSES, NUM_FEATURES, validate
from network import ANN_Model

SESSION_ID = os.environ.get("FL_SESSION_ID", "")
SERVER_URL = os.environ.get("FL_SERVER_URL", "http://127.0.0.1:8000")
MAX_UPLOAD_BYTES = int(os.environ.get("FL_MAX_UPLOAD_MB", 300)) * 1024 * 1024
LOCAL_EPOCHS = os.environ.get("FL_LOCAL_EPOCHS", "5")
LR = os.environ.get("FL_LR", "0.001")
NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", 5))
CHECKPOINT_EVERY = int(os.environ.get("FL_CHECKPOINT_EVERY", 25))
BASE_ALPHA = float(os.environ.get("FL_BASE_ALPHA", 0.5))
MAX_STALENESS = int(os.environ.get("FL_MAX_STALENESS", 20))
EVAL_EVERY = int(os.environ.get("FL_EVAL_EVERY", 5))

app = FastAPI(title="Async Federated Learning Server")
lock = threading.Lock()

# The server holds no data: the global model starts from random weights and
# is evaluated only through confusion matrices that clients compute locally.
torch.manual_seed(SEED)
model = ANN_Model(input_features=NUM_FEATURES, out_features=NUM_CLASSES)

state = {"version": 0, "accepted": 0, "rejected": 0, "history": [], "client_cms": {},
         "eval_reports": {}}

if os.environ.get("FL_RESUME") == "1":
    model.load_state_dict(torch.load(os.path.join(HERE, "global_model_async.pt"), map_location="cpu"))


class Update(BaseModel):
    client_id: int
    base_version: int
    num_samples: int
    weights: dict
    val_confusion_matrix: list | None = None  # global model scored on the client's local validation data


class EvalReport(BaseModel):
    reporter: str
    version: int
    confusion_matrix: list  # final model scored on all of the reporter's local data


def _summarize(cm_list):
    """Overall accuracy plus macro precision/recall/F1 from a (summed) confusion matrix."""
    cm = np.array(cm_list, dtype=float)
    tp = np.diag(cm)
    predicted, actual = cm.sum(axis=0), cm.sum(axis=1)
    # macro averages over classes; classes absent from the data are skipped
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = actual > 0
    return {
        "samples": int(cm.sum()),
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _log_metrics():
    if not state["client_cms"]:
        return
    entry = {
        "update": state["accepted"],
        "version": state["version"],
        **_summarize(np.sum(list(state["client_cms"].values()), axis=0)),
    }
    state["history"].append(entry)
    print(
        f"Update {entry['update']:03d} | v{entry['version']} | acc {entry['accuracy']:.4f} "
        f"| recall {entry['recall']:.4f} | f1 {entry['f1']:.4f}",
        flush=True,
    )


CHECKPOINT_PATH = os.path.join(HERE, "global_model_async.pt")


def _save_checkpoint():
    torch.save(model.state_dict(), CHECKPOINT_PATH)


@app.get("/model")
def get_model(x_session_id: str = Header(default="")):
    _check_session(x_session_id)
    with lock:
        return {
            "version": state["version"],
            "weights": state_to_json(model.state_dict()),
        }


@app.post("/checkpoint")
def checkpoint(x_session_id: str = Header(default="")):
    _check_session(x_session_id)
    with lock:
        _save_checkpoint()
        _log_metrics()
        return {"saved": CHECKPOINT_PATH, "version": state["version"]}


@app.post("/update")
def post_update(update: Update, x_session_id: str = Header(default="")):
    _check_session(x_session_id)
    with lock:
        staleness = state["version"] - update.base_version
        if staleness < 0:
            raise HTTPException(400, "base_version is ahead of the server")
        if staleness > MAX_STALENESS:
            state["rejected"] += 1
            return {"accepted": False, "reason": "too stale", "version": state["version"]}

        if update.val_confusion_matrix is not None:
            state["client_cms"][update.client_id] = np.array(update.val_confusion_matrix)

        merged, alpha = async_merge(
            model.state_dict(), state_from_json(update.weights), BASE_ALPHA, staleness
        )
        model.load_state_dict(merged)
        state["version"] += 1
        state["accepted"] += 1

        if state["accepted"] % EVAL_EVERY == 0:
            _log_metrics()
        if state["accepted"] % CHECKPOINT_EVERY == 0:
            _save_checkpoint()

        return {"accepted": True, "alpha": alpha, "staleness": staleness,
                "version": state["version"]}


joined = {"next_id": 0, "clients": {}}


def _check_session(session_id):
    if not SESSION_ID or not hmac.compare_digest(session_id or "", SESSION_ID):
        raise HTTPException(404, "unknown session id")


@app.post("/register/{session_id}")
def register(session_id: str):
    """Private mode: join without uploading data; the client trains on its own machine."""
    _check_session(session_id)
    with lock:
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

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "could not parse the file as CSV")
    try:
        df = validate(df)
    except ValueError as e:
        raise HTTPException(400, str(e))
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


@app.post("/eval-report")
def post_eval_report(report: EvalReport, x_session_id: str = Header(default="")):
    """Final evaluation: a client scored the model on its own data and shares only the counts."""
    _check_session(x_session_id)
    cm = np.array(report.confusion_matrix)
    if cm.shape != (NUM_CLASSES, NUM_CLASSES):
        raise HTTPException(400, f"confusion_matrix must be {NUM_CLASSES}x{NUM_CLASSES}")
    with lock:
        state["eval_reports"][report.reporter] = {"version": report.version, "cm": cm}
        return {"recorded": True, "reporters": len(state["eval_reports"])}


@app.get("/eval-report")
def get_eval_report(x_session_id: str = Header(default="")):
    """Aggregate of all clients' final-evaluation reports (only reports for the newest version are combined)."""
    _check_session(x_session_id)
    with lock:
        reports = state["eval_reports"]
        if not reports:
            return {"reporters": 0, "overall": None, "per_client": {}}
        newest = max(r["version"] for r in reports.values())
        used = {k: r for k, r in reports.items() if r["version"] == newest}
        return {
            "model_version": newest,
            "reporters": len(used),
            "overall": _summarize(np.sum([r["cm"] for r in used.values()], axis=0)),
            "per_client": {k: _summarize(r["cm"]) | {"version": r["version"]} for k, r in used.items()},
            "stale_reporters": [k for k, r in reports.items() if r["version"] != newest],
        }


@app.get("/status")
def status():
    with lock:
        return {
            "version": state["version"],
            "accepted": state["accepted"],
            "rejected": state["rejected"],
            "clients": dict(joined["clients"]),
            "latest": state["history"][-1] if state["history"] else None,
        }
