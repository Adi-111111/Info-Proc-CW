import pandas as pd
from datetime import datetime
import os


INPUT_FILE = "metrics/gesture_metrics.csv"
OUTPUT_DIR = "metrics/reports"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    lat = df["latency_ms"]

    summary = {
        "gestures": len(lat),
        "mean_ms": lat.mean(),
        "median_ms": lat.median(),
        "p95_ms": lat.quantile(0.95),
        "p99_ms": lat.quantile(0.99),
        "std_ms": lat.std()
    }

    by_shape = df.groupby("shape")["latency_ms"].mean()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = f"{OUTPUT_DIR}/latency_report_{timestamp}.txt"
    csv_summary_path = f"{OUTPUT_DIR}/latency_summary_{timestamp}.csv"

    # -----------------------------
    # Write text report
    # -----------------------------
    with open(report_path, "w") as f:

        f.write("Gesture Recognition Latency Report\n")
        f.write("=================================\n\n")

        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Total gestures: {summary['gestures']}\n\n")

        f.write("Overall Latency Statistics (ms)\n")
        f.write("-------------------------------\n")
        f.write(f"Mean:   {summary['mean_ms']:.2f}\n")
        f.write(f"Median: {summary['median_ms']:.2f}\n")
        f.write(f"P95:    {summary['p95_ms']:.2f}\n")
        f.write(f"P99:    {summary['p99_ms']:.2f}\n")
        f.write(f"Std:    {summary['std_ms']:.2f}\n\n")

        f.write("Average Latency by Shape\n")
        f.write("------------------------\n")

        for shape, val in by_shape.items():
            f.write(f"{shape}: {val:.2f} ms\n")

    # -----------------------------
    # Write machine-readable CSV
    # -----------------------------
    pd.DataFrame([summary]).to_csv(csv_summary_path, index=False)

    print(f"Report written to: {report_path}")
    print(f"Summary CSV written to: {csv_summary_path}")


if __name__ == "__main__":
    main()