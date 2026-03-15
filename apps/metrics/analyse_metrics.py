import json
import pandas as pd
from pathlib import Path


LOG_FILE = Path("apps/metrics/events.jsonl")


# --------------------------------------------------
# Load event log
# --------------------------------------------------
def load_events():

    rows = []

    with open(LOG_FILE) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if df.empty:
        print("No events found.")
        exit()

    return df


# --------------------------------------------------
# Build gesture timelines
# --------------------------------------------------
def build_timelines(df):

    # pivot events so each object_id becomes a row
    timelines = df.pivot_table(
        index="object_id",
        columns="event",
        values="timestamp",
        aggfunc="first"
    )

    return timelines.reset_index()


# --------------------------------------------------
# Compute latency metrics
# --------------------------------------------------
def compute_metrics(t):

    metrics = pd.DataFrame()

    metrics["object_id"] = t["object_id"]

    # recognition latency
    metrics["recognition_latency"] = (
        t["shape_fully_classified"] - t["gesture_end"]
    )

    # network latency
    metrics["network_latency"] = (
        t["whiteboard_receive"] - t["send_to_network"]
    )

    # end-to-end latency
    metrics["end_to_end_latency"] = (
        t["render_done"] - t["gesture_end"]
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

        m = metrics[col].dropna()

        if len(m) == 0:
            continue

        print(col)
        print("count :", len(m))
        print("mean  :", round(m.mean()*1000, 2), "ms")
        print("median:", round(m.median()*1000, 2), "ms")
        print("p95   :", round(m.quantile(0.95)*1000, 2), "ms")
        print("max   :", round(m.max()*1000, 2), "ms")
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
# Main
# --------------------------------------------------
def main():

    df = load_events()

    timelines = build_timelines(df)

    metrics = compute_metrics(timelines)

    print_summary(metrics)

    print_timelines(metrics)


if __name__ == "__main__":
    main()