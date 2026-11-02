"""Launcher for asynchronous federated learning.

Starts the FastAPI server, spawns the client processes, waits for training to
finish, then prints the final metrics. (Synchronous version: sync_main.py)
"""
import os
import subprocess
import sys
import time

import requests

NUM_CLIENTS = 5
MAX_UPDATES = 100  # same total number of local trainings as sync: 20 rounds x 5 clients
LOCAL_EPOCHS = 5
LR = 0.01
MAX_DELAY = 3.0
HOST, PORT = "127.0.0.1", 8000

HERE = os.path.dirname(os.path.abspath(__file__))
URL = f"http://{HOST}:{PORT}"


def wait_for_server(proc, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError("Server exited early")
        try:
            requests.get(f"{URL}/status", timeout=1)
            return
        except requests.RequestException:
            time.sleep(0.5)
    raise TimeoutError("Server did not start in time")


def main():
    env = dict(
        os.environ,
        FL_NUM_CLIENTS=str(NUM_CLIENTS),
        FL_MAX_UPDATES=str(MAX_UPDATES),
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", HOST,
         "--port", str(PORT), "--log-level", "warning"],
        cwd=HERE, env=env,
    )
    clients = []
    try:
        wait_for_server(server)
        for i in range(NUM_CLIENTS):
            clients.append(subprocess.Popen(
                [sys.executable, "async_client.py", "--id", str(i),
                 "--num-clients", str(NUM_CLIENTS), "--url", URL,
                 "--epochs", str(LOCAL_EPOCHS), "--lr", str(LR),
                 "--max-delay", str(MAX_DELAY)],
                cwd=HERE,
            ))
        for c in clients:
            c.wait()

        status = requests.get(f"{URL}/status", timeout=10).json()
        latest = status["latest"]
        print(f"\nAccepted updates: {status['accepted']} | rejected: {status['rejected']}")
        if latest:
            print("Final accuracy:", round(latest["accuracy"], 4),
                  "| recall:", round(latest["recall"], 4))
            print("Confusion matrix (rows=actual, cols=predicted):\n", latest["confusion_matrix"])
    finally:
        for c in clients:
            if c.poll() is None:
                c.terminate()
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
