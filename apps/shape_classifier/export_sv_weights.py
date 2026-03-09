import json
from pathlib import Path

EXPORT_PATH = Path(__file__).resolve().parent / "weights_export" / "mlp_quantized.json"
OUT_PATH = Path(__file__).resolve().parents[2] / "hardware" / "mlp_weights.svh"

def main():
    with open(EXPORT_PATH, "r") as f:
        data = json.load(f)

    fc1_w = data["fc1_weight"]   # [16][70]
    fc1_b = data["fc1_bias"]     # [16]
    fc2_w = data["fc2_weight"]   # [5][16]
    fc2_b = data["fc2_bias"]     # [5]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        f.write("// Auto-generated quantized MLP weights\n\n")

        for j in range(16):
            for i in range(70):
                f.write(f"initial w1[{j}][{i}] = 8'sd{fc1_w[j][i]};\n")

        f.write("\n")
        for j in range(16):
            f.write(f"initial b1[{j}] = 32'sd{fc1_b[j]};\n")

        f.write("\n")
        for o in range(5):
            for j in range(16):
                f.write(f"initial w2[{o}][{j}] = 8'sd{fc2_w[o][j]};\n")

        f.write("\n")
        for o in range(5):
            f.write(f"initial b2[{o}] = 32'sd{fc2_b[o]};\n")

    print(f"Saved SV weights to: {OUT_PATH}")

if __name__ == "__main__":
    main()