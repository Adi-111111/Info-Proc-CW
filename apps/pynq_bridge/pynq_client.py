"""
PYNQ → Whiteboard bridge client

Receives shape JSON from test.py over UDP
and forwards it to the whiteboard server
"""

import socket
import json
import socketio
import uuid
# from apps.metrics.logger import log_event
import time

# CONFIG
SERVER_URL = "http://13.40.61.155:5000"

BUFFER_SIZE = 65535

BRIDGE_IP = "127.0.0.1"
BRIDGE_PORT = 5010


# SOCKET.IO CLIENT
sio = socketio.Client(logger=True, engineio_logger=True)
@sio.event
def connect():
    sio.emit("register_pynq")
    print("[SERVER] Connected to server")

@sio.event
def disconnect():
    print("[SERVER] Disconnected from server")


# UDP SOCKET - COMMUNICATION WITH TEST.PY
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #why are we using DGRAM? what are the other options?
sock.bind((BRIDGE_IP, BRIDGE_PORT)) #which IP address and port program should listen on for incoming packets.

print("\n[UDP] Listening for shapes from laptop")
print("[UDP] Port:", BRIDGE_PORT)


# MAIN - Only start the UDP listening loop if this file is executed directly - might need to change this for bash scripts. 
if __name__ == "__main__": 
    print("\n[CLIENT] Connecting to whiteboard server...")
    sio.connect(SERVER_URL)
    try:
        while True:
            print("\n-----------------------------")
            print("[UDP] Waiting for packet...")

            data, addr = sock.recvfrom(BUFFER_SIZE)#receive data from a certain address over the socket with a max packet size of BUFFER_SIZE
            print("[UDP] Packet received from:", addr)
            print("[UDP] Raw bytes:", data)

            try:

                shape = json.loads(data.decode()) #JSON string -> Python Object
                print("[UDP] JSON decoded successfully")
            except Exception as e:
                print("[ERROR] JSON decode failed:", e)
                continue

            print("\n[UDP] Shape received:")
            print(json.dumps(shape, indent=2)) #Python Object -> JSON String


            # Expected shape formats (examples):
            #   circle:    {"type": "circle", "cx": 150.5, "cy": 200.25, "r": 50.75}
            #   rectangle: {"type": "rectangle", "corners": [[x,y],[x,y],[x,y],[x,y]]}
            #   triangle:  {"type": "triangle", "corners": [[x1,y1],[x2,y2],[x3,y3]]}
            #   stroke:    {"type": "stroke", "points": [[x,y], ...]}

            # Check whether shape contains a type (has it been classified or not?)
            if "type" not in shape:
                print("[ERROR] Missing 'type' field")
                continue

            # Add object_id if missing
            if "object_id" not in shape:
                shape["object_id"] = f"obj_{uuid.uuid4().hex[:8]}" 
                print("[CLIENT] Generated object_id:", shape["object_id"])

            print("\n[CLIENT] Sending to server:")
            print(json.dumps(shape, indent=2)) #Python Object -> JSON string

            # send_to_network = time.perf_counter() # metrics
            # log_event( 
            #    "send_to_network",
            #    shape["object_id"],
            #    timestamp = send_to_network,
            #    component="pynq_bridge"
            #)

            sio.emit("pynq_event", shape)

            print("[CLIENT] Shape forwarded successfully")

    except KeyboardInterrupt:

        print("\n[CLIENT] Stopping client...")

    finally:

        sock.close()
        sio.disconnect()

        print("[CLIENT] Shutdown complete")