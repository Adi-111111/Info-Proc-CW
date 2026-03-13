import json
from pathlib import Path

EXPORT_PATH = Path(__file__).resolve().parent / "weights_export" / "mlp_quantized.json"
OUT_PATH = Path(__file__).resolve().parents[2] / "hardware" / "sv" / "mlp_weights.svh"


def sv_signed_literal(value, width):
    value = int(value)
    if value < 0:
        return f"-{width}'sd{abs(value)}"
    return f"{width}'sd{value}"


def main():
    with open(EXPORT_PATH, "r") as f:
        data = json.load(f)

    fc1_w = data["fc1_weight"]
    fc1_b = data["fc1_bias"]
    fc2_w = data["fc2_weight"]
    fc2_b = data["fc2_bias"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        f.write("// Auto-generated quantized MLP weights\n\n")

        for j in range(16):
            for i in range(70):
                lit = sv_signed_literal(fc1_w[j][i], 8)
                f.write(f"initial w1[{j}][{i}] = {lit};\n")

        f.write("\n")
        for j in range(16):
            lit = sv_signed_literal(fc1_b[j], 32)
            f.write(f"initial b1[{j}] = {lit};\n")

        f.write("\n")
        for o in range(5):
            for j in range(16):
                lit = sv_signed_literal(fc2_w[o][j], 8)
                f.write(f"initial w2[{o}][{j}] = {lit};\n")

        f.write("\n")
        for o in range(5):
            lit = sv_signed_literal(fc2_b[o], 32)
            f.write(f"initial b2[{o}] = {lit};\n")

    print(f"Saved SV weights to: {OUT_PATH}")


if __name__ == "__main__":
    main()