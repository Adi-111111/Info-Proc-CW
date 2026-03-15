import json
import time
import os

LOG_FILE = "metrics/events.jsonl"

os.makedirs("metrics", exist_ok=True)


def log_event(event, gesture_id):

    entry = {
        "gesture_id": gesture_id,
        "event": event,
        "timestamp": time.perf_counter()
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")