import json
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

from preprocess import preprocess_to_vector

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

HOLDOUT_ROOT = Path(__file__).resolve().parents[2] / "holdout_test"
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "tiny_mlp.pth"


class TinyMLP(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=16, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_holdout():
    X = []
    y = []

    for class_name in CLASS_NAMES:
        files = sorted(glob.glob(str(HOLDOUT_ROOT / class_name / "*.json")))
        print(f"{class_name}: {len(files)} files")

        for filename in files:
            with open(filename, "r") as f:
                data = json.load(f)

            stroke = data["stroke"]
            vec = preprocess_to_vector(stroke, num_points=32, min_distance=2.0)
            if vec is None:
                continue

            X.append(vec)
            y.append(CLASS_TO_IDX[class_name])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    X, y = load_holdout()
    print(f"\nTotal holdout samples: {len(X)}")

    model = TinyMLP(input_dim=70, hidden_dim=16, num_classes=5)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    X_t = torch.tensor(X)
    with torch.no_grad():
        logits = model(X_t)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y, preds)
    print("\nHoldout Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()