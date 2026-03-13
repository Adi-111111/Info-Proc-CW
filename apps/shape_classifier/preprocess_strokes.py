import json
import glob
import os
import matplotlib.pyplot as plt

from preprocess import preprocess, flatten_points

files = sorted(glob.glob("test_jsons/*.json"))

if not files:
    print("No JSON files found in test_jsons/")

for filename in files:
    with open(filename, "r") as f:
        data = json.load(f)

    raw = data["stroke"]
    print(f"\n{os.path.basename(filename)}: raw points = {len(raw)}")

    processed = preprocess(raw, num_points=32, min_distance=2.0)

    if processed is None:
        print("  skipped: not enough usable points after cleaning")
        continue

    feature_vector = flatten_points(processed)
    print(f"  processed points = {len(processed)}")
    print(f"  feature vector length = {len(feature_vector)}")

    raw_x = [p[0] for p in raw]
    raw_y = [p[1] for p in raw]

    proc_x = [p[0] for p in processed]
    proc_y = [p[1] for p in processed]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(raw_x, raw_y, marker="o")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title("Raw stroke")

    plt.subplot(1, 2, 2)
    plt.plot(proc_x, proc_y, marker="o")
    plt.axis("equal")
    plt.title("Processed stroke")

    plt.suptitle(os.path.basename(filename))
    plt.show()