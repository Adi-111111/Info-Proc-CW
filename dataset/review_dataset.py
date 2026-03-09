import json
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt

# Change this to whichever class you want to review
CLASS_NAME = "freehand"

files = sorted(glob.glob(f"dataset/{CLASS_NAME}/*.json"))

if not files:
    print(f"No files found in dataset/{CLASS_NAME}/")
    raise SystemExit

print(f"Reviewing {len(files)} files in dataset/{CLASS_NAME}/")
print("Controls: k = keep, d = delete, q = quit")

idx = 0

while idx < len(files):
    filename = files[idx]

    with open(filename, "r") as f:
        data = json.load(f)

    stroke = data["stroke"]
    xs = [p[0] for p in stroke]
    ys = [p[1] for p in stroke]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, marker="o")
    ax.invert_yaxis()
    ax.axis("equal")
    ax.set_title(os.path.basename(filename))
    fig.suptitle("k = keep | d = delete | q = quit")

    action = {"key": None}

    def on_key(event):
        if event.key in ["k", "d", "q"]:
            action["key"] = event.key
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if action["key"] == "k":
        print(f"[KEEP]   {filename}")
        idx += 1

    elif action["key"] == "d":
        try:
            Path(filename).unlink()
            print(f"[DELETE] {filename}")
        except Exception as e:
            print(f"[ERROR] Could not delete {filename}: {e}")
        idx += 1

    elif action["key"] == "q":
        print("Quitting review.")
        break

    else:
        print("No valid key pressed. Use k, d, or q.")