import argparse
import random
import time

import requests
import torch

from client import Client
from common import load_data, state_from_json, state_to_json
from network import ANN_Model


def run(args):
    torch.set_num_threads(1)
    rng = random.Random(args.id)

    prep, shards, _ = load_data(args.num_clients)
    X, y = shards[args.id]
    client = Client(args.id, X, y)
    model = ANN_Model(input_features=prep.num_features)
    session = requests.Session()

    pushes = 0
    while True:
        resp = session.get(f"{args.url}/model", timeout=30).json()
        if resp["done"]:
            break
        model.load_state_dict(state_from_json(resp["weights"]))
        base_version = resp["version"]

        weights, n, loss = client.train(model, args.epochs, args.lr)

        time.sleep(rng.uniform(0, args.max_delay))

        result = session.post(
            f"{args.url}/update",
            json={
                "client_id": args.id,
                "base_version": base_version,
                "num_samples": n,
                "weights": state_to_json(weights),
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
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-delay", type=float, default=3.0,
                   help="max random sleep (s) before pushing, simulates slow devices")
    run(p.parse_args())
