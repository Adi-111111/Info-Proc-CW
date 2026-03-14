import socket
import json
import time
from pynq import Overlay

# -----------------------------
# Load overlay
# -----------------------------
ol = Overlay("/home/xilinx/jupyter_notebooks/mlp/mlp.bit")
mlp = ol.mlp_axi_v1_0_MLP_AXI_0

CLASS_NAMES = ["circle", "rectangle", "triangle", "line", "freehand"]

# -----------------------------
# UDP setup
# -----------------------------
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[udp] listening on {UDP_IP}:{UDP_PORT}")

# default reply destination if packet doesn't include one
DEFAULT_REPLY_IP = "192.168.2.1"
DEFAULT_REPLY_PORT = 5006

# -----------------------------
# AXI helpers
# -----------------------------
def pack_features_to_regs(features):
    if len(features) != 70:
        raise ValueError(f"Expected 70 features, got {len(features)}")

    byte_vals = [(int(x) + 256) % 256 for x in features]

    regs = []
    for i in range(0, 70, 4):
        chunk = byte_vals[i:i+4]
        while len(chunk) < 4:
            chunk.append(0)
        word = (
            (chunk[0] << 0)  |
            (chunk[1] << 8)  |
            (chunk[2] << 16) |
            (chunk[3] << 24)
        )
        regs.append(word)

    while len(regs) < 18:
        regs.append(0)

    return regs

def mlp_write_features(features):
    regs = pack_features_to_regs(features)
    for i, word in enumerate(regs):
        addr = 0x04 + 4 * i
        mlp.write(addr, word)

def mlp_start_hold():
    mlp.write(0x00, 0x1)

def mlp_stop():
    mlp.write(0x00, 0x0)

def mlp_status():
    return mlp.read(0x00)

def mlp_done():
    return (mlp_status() >> 8) & 0x1

def mlp_predict(features, timeout_s=1.0):
    mlp_write_features(features)
    mlp_start_hold()

    t0 = time.time()
    while mlp_done() == 0:
        if time.time() - t0 > timeout_s:
            mlp_stop()
            raise TimeoutError("Timed out waiting for accelerator")

    status = mlp_status()
    cid = (status >> 9) & 0x7
    label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"unknown({cid})"

    mlp_stop()
    return cid, label, status

# -----------------------------
# Main loop
# -----------------------------
while True:
    data, addr = sock.recvfrom(65535)

    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        print("[error] invalid JSON:", e)
        continue

    if "features" not in payload:
        print("[error] no features field")
        continue

    features = payload["features"]
    reply_ip = payload.get("reply_ip", DEFAULT_REPLY_IP)
    reply_port = int(payload.get("reply_port", DEFAULT_REPLY_PORT))

    print(f"\n[udp] received {len(features)} features from {addr}")
    print("[udp] first 10 features:", features[:10])

    try:
        cid, label, status = mlp_predict(features)

        reply = {
            "class_id": cid,
            "label": label,
            "status": int(status)
        }

        sock.sendto(json.dumps(reply).encode("utf-8"), (reply_ip, reply_port))

        print("[fpga] predicted:", label)
        print("[fpga] class_id:", cid)
        print("[fpga] status:", hex(status))
        print(f"[udp] replied to {(reply_ip, reply_port)}")

    except Exception as e:
        err = {
            "error": str(e)
        }
        sock.sendto(json.dumps(err).encode("utf-8"), (reply_ip, reply_port))
        print("[error] inference failed:", e)