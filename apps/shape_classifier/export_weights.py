import json
from pathlib import Path

import torch
import torch.nn as nn


CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "tiny_mlp.pth"
OUT_DIR = Path(__file__).resolve().parent / "weights_export"
OUT_DIR.mkdir(exist_ok=True)


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


def quantize_tensor(t, scale):
    q = torch.round(t * scale).to(torch.int32)
    q = torch.clamp(q, -128, 127)
    return q


def main():
    model = TinyMLP(input_dim=70, hidden_dim=16, num_classes=5)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    fc1 = model.net[0]
    fc2 = model.net[2]

    fc1_w = fc1.weight.detach().cpu()   # [16, 70]
    fc1_b = fc1.bias.detach().cpu()     # [16]
    fc2_w = fc2.weight.detach().cpu()   # [5, 16]
    fc2_b = fc2.bias.detach().cpu()     # [5]

    # Fixed-point scale
    weight_scale = 64
    bias_scale = 64

    fc1_w_q = quantize_tensor(fc1_w, weight_scale)
    fc1_b_q = torch.round(fc1_b * bias_scale).to(torch.int32)
    fc2_w_q = quantize_tensor(fc2_w, weight_scale)
    fc2_b_q = torch.round(fc2_b * bias_scale).to(torch.int32)

    export = {
        "meta": {
            "input_dim": 70,
            "hidden_dim": 16,
            "output_dim": 5,
            "class_names": CLASS_NAMES,
            "weight_scale": weight_scale,
            "bias_scale": bias_scale,
        },
        "fc1_weight": fc1_w_q.tolist(),
        "fc1_bias": fc1_b_q.tolist(),
        "fc2_weight": fc2_w_q.tolist(),
        "fc2_bias": fc2_b_q.tolist(),
    }

    out_json = OUT_DIR / "mlp_quantized.json"
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)

    print(f"Saved quantized weights to: {out_json}")
    print("fc1_weight shape:", fc1_w_q.shape)
    print("fc1_bias shape:", fc1_b_q.shape)
    print("fc2_weight shape:", fc2_w_q.shape)
    print("fc2_bias shape:", fc2_b_q.shape)


if __name__ == "__main__":
    main()