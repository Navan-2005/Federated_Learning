"""Launcher: starts the federated server, creates a unique session id and waits.

Participants join by uploading their CSV to  POST /join/<session_id>.
Training finishes after MAX_UPDATES accepted updates; the final model is saved
to global_model_async.pt.  (Synchronous simulation: sync_main.py)
"""
import os
import subprocess
import sys
import time
import uuid

import requests

MAX_UPDATES = int(os.environ.get("FL_MAX_UPDATES", 100))
HOST = os.environ.get("FL_HOST", "0.0.0.0")  # 0.0.0.0 lets other machines reach the server
PORT = int(os.environ.get("FL_PORT", 8000))
PUBLIC_HOST = os.environ.get("FL_PUBLIC_HOST", "127.0.0.1")  # address shown in the join URL

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_URL = f"http://127.0.0.1:{PORT}"


def wait_for_server(proc, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError("Server exited early")
        try:
            requests.get(f"{LOCAL_URL}/status", timeout=1)
            return
        except requests.RequestException:
            time.sleep(0.5)
    raise TimeoutError("Server did not start in time")


def main():
    session_id = uuid.uuid4().hex
    public_url = f"http://{PUBLIC_HOST}:{PORT}"
    env = dict(
        os.environ,
        FL_SESSION_ID=session_id,
        FL_MAX_UPDATES=str(MAX_UPDATES),
        FL_SERVER_URL=LOCAL_URL,  # server-side workers talk to the server locally
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", HOST,
         "--port", str(PORT), "--log-level", "warning"],
        cwd=HERE, env=env,
    )
    try:
        wait_for_server(server)
        print("Global model initialised (random weights, version 0).")
        print(f"Session id : {session_id}")
        print(f"Join URL   : POST {public_url}/join/{session_id}   (multipart field 'file' = CSV)")
        print(f'Example    : curl -F "file=@my_data.csv" {public_url}/join/{session_id}')
        print("            (upload mode: the server receives the CSV)")
        print("Private    : python real_client.py "
              f"--session-id {session_id} --data my_data.csv --url {public_url}")
        print("            (private mode: data stays on the participant's machine)")
        print(f"Training ends after {MAX_UPDATES} accepted updates. Ctrl+C to stop.\n", flush=True)

        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited unexpectedly")
            status = requests.get(f"{LOCAL_URL}/status", timeout=10).json()
            if status["done"]:
                break
            time.sleep(2)

        time.sleep(2)  # let finishing workers exit
        latest = status["latest"]
        print(f"\nAccepted updates: {status['accepted']} | rejected: {status['rejected']}")
        if latest:
            print("Final accuracy:", round(latest["accuracy"], 4),
                  "| recall:", round(latest["recall"], 4))
            print("Confusion matrix (rows=actual, cols=predicted):\n", latest["confusion_matrix"])
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
