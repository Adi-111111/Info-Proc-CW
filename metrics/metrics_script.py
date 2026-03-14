import time
import csv
import os


class MetricsLogger:
    def __init__(self, file="metrics/gesture_metrics.csv"):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        self.file = file

        if not os.path.exists(file):
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "gesture_id",
                    "gesture_end_time",
                    "classification_time",
                    "latency_ms",
                    "shape"
                ])

    def log(self, gesture_id, gesture_end, classify_time, shape):

        latency = (classify_time - gesture_end) * 1000

        with open(self.file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                gesture_id,
                gesture_end,
                classify_time,
                latency,
                shape
            ])