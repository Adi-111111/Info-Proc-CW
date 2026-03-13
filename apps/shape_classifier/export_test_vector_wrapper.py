import json
from pathlib import Path

from preprocess import preprocess_to_vector

OUT_PATH = Path(__file__).resolve().parents[2] / "hardware" / "sv" / "test_vector_wrapper.svh"

# CHANGE THIS to a real sample you want to test
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "dataset" / "circle" / "circle_1772893882719.json"


def sv_signed_literal(value, width):
    value = int(value)
    if value < 0:
        return f"-{width}'sd{abs(value)}"
    return f"{width}'sd{value}"


def quantize_input(vec, scale=64):
    q = []
    for x in vec:
        val = int(round(x * scale))
        val = max(-128, min(127, val))
        q.append(val)
    return q


def main():
    with open(SAMPLE_PATH, "r") as f:
        data = json.load(f)

    stroke = data["stroke"]
    vec = preprocess_to_vector(stroke, num_points=32, min_distance=2.0)

    if vec is None:
        raise ValueError("Preprocessing returned None")

    if len(vec) != 70:
        raise ValueError(f"Expected 70 features, got {len(vec)}")

    q = quantize_input(vec, scale=64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("// Auto-generated wrapper test vector\n")
        for i, v in enumerate(q):
            lit = sv_signed_literal(v, 8)
            lo = i * 8
            hi = lo + 7
            f.write(f"        input_bus[{hi}:{lo}] = {lit};\n")

    print(f"Saved wrapper test vector to: {OUT_PATH}")
    print("First 10 quantized values:", q[:10])


if __name__ == "__main__":
    main()