import json
import time
import os

LOG_FILE = "apps/metrics/events.jsonl"

os.makedirs("apps/metrics", exist_ok=True)


def log_event(event, object_id, timestamp=None, component=None):

    if timestamp is None:
        timestamp = time.perf_counter()

    entry = {
        "object_id": object_id,
        "event": event,
        "timestamp": timestamp,
        "component": component
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")