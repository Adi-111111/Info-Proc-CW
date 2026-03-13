import json
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocess import preprocess_to_vector

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def load_dataset():
    X = []
    y = []

    for class_name in CLASS_NAMES:
        class_dir = DATASET_ROOT / class_name
        files = sorted(glob.glob(str(class_dir / "*.json")))

        print(f"{class_name}: {len(files)} files")

        for filename in files:
            with open(filename, "r") as f:
                data = json.load(f)

            if "stroke" not in data:
                continue

            stroke = data["stroke"]
            vector = preprocess_to_vector(stroke, num_points=32, min_distance=2.0)

            if vector is None:
                continue

            X.append(vector)
            y.append(CLASS_TO_IDX[class_name])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


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


def main():
    X, y = load_dataset()

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    model = TinyMLP(input_dim=70, hidden_dim=16, num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                preds = torch.argmax(test_logits, dim=1)
                acc = accuracy_score(y_test.numpy(), preds.numpy())

            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        preds = torch.argmax(test_logits, dim=1)

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    print("\nFinal Test Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), preds.numpy(), target_names=CLASS_NAMES))

    out_dir = Path(__file__).resolve().parent / "weights"
    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "tiny_mlp.pth"
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved model weights to: {model_path}")


if __name__ == "__main__":
    main()