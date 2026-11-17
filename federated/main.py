"""Launcher: starts the federated server, creates a unique session id and keeps running.

Participants join with real_client.py (private mode) or by uploading a CSV to
POST /join/<session_id>. Training is continuous: the global model keeps
improving as clients push updates, and is checkpointed to global_model_async.pt
every FL_CHECKPOINT_EVERY accepted updates. Press Ctrl+C to stop; a final
checkpoint is saved first.
"""
import os
import subprocess
import sys
import time
import uuid

import requests

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
    resume = "--resume" in sys.argv  # reload global_model_async.pt instead of random init
    session_id = os.environ.get("FL_SESSION_ID") or uuid.uuid4().hex
    public_url = f"http://{PUBLIC_HOST}:{PORT}"
    env = dict(
        os.environ,
        FL_SESSION_ID=session_id,
        FL_RESUME="1" if resume else "0",
        FL_SERVER_URL=LOCAL_URL,  # server-side workers talk to the server locally
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", HOST,
         "--port", str(PORT), "--log-level", "warning"],
        cwd=HERE, env=env,
    )
    try:
        wait_for_server(server)
        print("Global model loaded from global_model_async.pt." if resume
              else "Global model initialised (random weights, version 0).")
        print(f"Session id : {session_id}")
        print(f"Join URL   : POST {public_url}/join/{session_id}   (multipart field 'file' = CSV)")
        print(f'Example    : curl -F "file=@my_data.csv" {public_url}/join/{session_id}')
        print("            (upload mode: the server receives the CSV)")
        print("Private    : python real_client.py "
              f"--session-id {session_id} --data my_data.csv --url {public_url}")
        print("            (private mode: data stays on the participant's machine)")
        print("Training is continuous. Press Ctrl+C to save a checkpoint and stop.\n", flush=True)

        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited unexpectedly")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if server.poll() is None:
            try:
                requests.post(f"{LOCAL_URL}/checkpoint",
                              headers={"X-Session-Id": session_id}, timeout=30)
                status = requests.get(f"{LOCAL_URL}/status", timeout=10).json()
                print(f"Accepted updates: {status['accepted']} | rejected: {status['rejected']}")
                latest = status["latest"]
                if latest:
                    print("Final accuracy:", round(latest["accuracy"], 4),
                          "| macro recall:", round(latest["recall"], 4))
                print("Model saved to global_model_async.pt")
            except requests.RequestException:
                pass
            server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
