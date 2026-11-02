import os
import threading

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aggregator import async_merge
from common import HERE, SEED, load_data, state_from_json, state_to_json
from evaluation import evaluate
from network import ANN_Model

NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", 5))
MAX_UPDATES = int(os.environ.get("FL_MAX_UPDATES", 100))
BASE_ALPHA = float(os.environ.get("FL_BASE_ALPHA", 0.5))
MAX_STALENESS = int(os.environ.get("FL_MAX_STALENESS", 20))
EVAL_EVERY = int(os.environ.get("FL_EVAL_EVERY", 5))

app = FastAPI(title="Async Federated Learning Server")
lock = threading.Lock()

prep, _, (X_test, y_test) = load_data(NUM_CLIENTS)
torch.manual_seed(SEED)
model = ANN_Model(input_features=prep.num_features)

state = {"version": 0, "accepted": 0, "rejected": 0, "history": []}


class Update(BaseModel):
    client_id: int
    base_version: int
    num_samples: int
    weights: dict


def _log_metrics():
    m = evaluate(model, X_test, y_test)
    entry = {
        "update": state["accepted"],
        "version": state["version"],
        "accuracy": float(m["accuracy"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "confusion_matrix": m["confusion_matrix"].tolist(),
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
def get_model():
    with lock:
        return {
            "version": state["version"],
            "done": _done(),
            "weights": state_to_json(model.state_dict()),
        }


@app.post("/update")
def post_update(update: Update):
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


@app.get("/status")
def status():
    with lock:
        return {
            "version": state["version"],
            "accepted": state["accepted"],
            "rejected": state["rejected"],
            "max_updates": MAX_UPDATES,
            "done": _done(),
            "latest": state["history"][-1] if state["history"] else None,
        }
