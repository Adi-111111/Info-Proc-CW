# ── MLP Benchmark (HW vs SW baseline) ────────────────────────────────────────
import time
import json
from pathlib import Path

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]

# ── SW baseline: pure Python integer MLP (mirrors fixed_point_ref.py) ─────────
def relu(x):
    return x if x > 0 else 0

def quantize_input(vec, scale=64):
    q = []
    for x in vec:
        val = int(round(x * scale))
        val = max(-128, min(127, val))
        q.append(val)
    return q

def mlp_infer_sw(x, params):
    fc1_w = params["fc1_weight"]
    fc1_b = params["fc1_bias"]
    fc2_w = params["fc2_weight"]
    fc2_b = params["fc2_bias"]

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
    return class_id

# ── HW inference (uses mlp_predict already defined in your notebook) ──────────
def benchmark_mlp(features_float, params, runs=100):
    """
    features_float : the raw float vector from preprocess_to_vector (length 70)
    params         : the loaded mlp_quantized.json dict
    runs           : number of repetitions for timing
    """
    x_q = quantize_input(features_float, scale=64)

    # ── SW timing ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(runs):
        sw_class = mlp_infer_sw(x_q, params)
    sw_ms = (time.perf_counter() - t0) / runs * 1000

    # ── HW timing ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(runs):
        hw_class, hw_label, _ = mlp_predict(x_q)
    hw_ms = (time.perf_counter() - t0) / runs * 1000

    speedup = sw_ms / hw_ms

    # ── Agreement check ───────────────────────────────────────────────────────
    agree = (sw_class == hw_class)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"{'Stage':<22} {'HW (ms)':>10}  {'SW (ms)':>10}  {'Speedup':>8}")
    print("-" * 56)
    print(f"  {'MLP inference':<20} {hw_ms:>10.4f}  {sw_ms:>10.4f}  {speedup:>7.2f}x")
    print()
    print(f"  SW predicted : {CLASS_NAMES[sw_class]} (class {sw_class})")
    print(f"  HW predicted : {hw_label} (class {hw_class})")
    print(f"  Agreement    : {'YES ✓' if agree else 'NO ✗ — MISMATCH'}")

    return hw_ms, sw_ms, agree

# ── Load weights and a test sample ───────────────────────────────────────────
WEIGHTS_PATH = Path("/home/xilinx/jupyter_notebooks/mlp/mlp_quantized.json")
SAMPLE_PATH  = Path("/home/xilinx/jupyter_notebooks/mlp/circle_1773088984993.json")

with open(WEIGHTS_PATH) as f:
    params = json.load(f)

with open(SAMPLE_PATH) as f:
    sample = json.load(f)

from preprocess import preprocess_to_vector

features = preprocess_to_vector(sample["stroke"], num_points=32, min_distance=2.0)

benchmark_mlp(features, params, runs=100)