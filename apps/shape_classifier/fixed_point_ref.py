import json
from pathlib import Path

from preprocess import preprocess_to_vector

EXPORT_PATH = Path(__file__).resolve().parent / "weights_export" / "mlp_quantized.json"

sample_file = Path(__file__).resolve().parents[2] / "dataset" / "triangle" / "triangle_1772899858701.json"

with open(sample_file, "r") as f:
    data = json.load(f)

sample_stroke = data["stroke"]

def relu(x):
    return x if x > 0 else 0


def quantize_input(vec, scale=64):
    q = []
    for x in vec:
        val = int(round(x * scale))
        val = max(-128, min(127, val))
        q.append(val)
    return q


def mlp_infer_int(x, params):
    fc1_w = params["fc1_weight"]   # [16][70]
    fc1_b = params["fc1_bias"]     # [16]
    fc2_w = params["fc2_weight"]   # [5][16]
    fc2_b = params["fc2_bias"]     # [5]

    hidden = []
    for j in range(16):
        acc = fc1_b[j]
        for i in range(70):
            acc += x[i] * fc1_w[j][i]
        hidden.append(relu(acc))

    outputs = []
    for k in range(5):
        acc = fc2_b[k]
        for j in range(16):
            acc += hidden[j] * fc2_w[k][j]
        outputs.append(acc)

    class_id = max(range(5), key=lambda k: outputs[k])
    return class_id, hidden, outputs


def main():
    with open(EXPORT_PATH, "r") as f:
        params = json.load(f)

    vec = preprocess_to_vector(sample_stroke, num_points=32, min_distance=2.0)
    x_q = quantize_input(vec, scale=64)

    class_id, hidden, outputs = mlp_infer_int(x_q, params)

    print("Quantized input length:", len(x_q))
    print("Hidden:", hidden)
    print("Outputs:", outputs)
    print("Predicted class:", params["meta"]["class_names"][class_id])


if __name__ == "__main__":
    main()