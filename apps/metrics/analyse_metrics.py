import json
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "events.jsonl"


# --------------------------------------------------
# Load event log
# --------------------------------------------------
def load_events():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    rows = []

    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if df.empty:
        print("No events found.")
        sys.exit()

    required_raw_cols = {"object_id", "event", "timestamp"}
    missing = required_raw_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in raw log: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


# --------------------------------------------------
# Build gesture timelines
# --------------------------------------------------
def build_timelines(df):
    timelines = df.pivot_table(
        index="object_id",
        columns="event",
        values="timestamp",
        aggfunc="first"
    ).reset_index()

    return timelines


# --------------------------------------------------
# Compute latency metrics
# --------------------------------------------------
def compute_metrics(t):

    metrics = pd.DataFrame()
    metrics["object_id"] = t["object_id"]

    def safe_diff(df, end_col, start_col):
        if end_col not in df.columns or start_col not in df.columns:
            return pd.Series(float("nan"), index=df.index)
        return df[end_col] - df[start_col]

    metrics["recognition_latency"] = safe_diff(
        t, "shape_fully_classified", "gesture_end"
    )

    metrics["network_latency"] = safe_diff(
        t, "whiteboard_receive", "send_to_network"
    )

    metrics["end_to_end_latency"] = safe_diff(
        t, "render_done", "gesture_end"
    )

    return metrics


# --------------------------------------------------
# Print summary statistics
# --------------------------------------------------
def print_summary(metrics):
    print("\n=== Metrics Summary ===\n")

    for col in [
        "recognition_latency",
        "network_latency",
        "end_to_end_latency"
    ]:
        m = pd.to_numeric(metrics[col], errors="coerce").dropna()

        if len(m) == 0:
            print(f"{col}: no complete data\n")
            continue

        print(col)
        print("count :", len(m))
        print("mean  :", round(m.mean() * 1000, 2), "ms")
        print("median:", round(m.median() * 1000, 2), "ms")
        print("p95   :", round(m.quantile(0.95) * 1000, 2), "ms")
        print("max   :", round(m.max() * 1000, 2), "ms")
        print()


# --------------------------------------------------
# Optional: print per-gesture debug table
# --------------------------------------------------
def print_timelines(metrics):
    print("\n=== Per Gesture Latencies ===\n")
    print(
        metrics[
            [
                "object_id",
                "recognition_latency",
                "network_latency",
                "end_to_end_latency"
            ]
        ].round(4)
    )


# --------------------------------------------------
# Debug helper
# --------------------------------------------------
def print_event_diagnostics(df, timelines):
    print("\n=== Raw Event Names ===\n")
    print(sorted(df["event"].dropna().unique().tolist()))

    print("\n=== Timeline Columns ===\n")
    print(timelines.columns.tolist())


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    df = load_events()
    timelines = build_timelines(df)

    print_event_diagnostics(df, timelines)

    metrics = compute_metrics(timelines)
    print_summary(metrics)
    print_timelines(metrics)


if __name__ == "__main__":
    main()