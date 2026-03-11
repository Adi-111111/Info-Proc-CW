"""
PYNQ → Whiteboard bridge client

Receives shape JSON from PYNQ over UDP
and forwards it to the whiteboard server
"""

import socket
import json
import socketio
import random

# CONFIG
SERVER_URL = "http://13.40.61.155:5000"
BOARD_ID = "board1" #how do we change this if we want to forward events to a different board?

PYNQ_PORT = 5006 #this shouldnt change
BUFFER_SIZE = 65535


# SOCKET.IO CLIENT
sio = socketio.Client(logger=True, engineio_logger=True)
@sio.event
def connect():
    print("\n[SERVER] Connected to whiteboard server")
    sio.emit("join_board", {
        "board_id": BOARD_ID
    })
    print("[SERVER] Joined board:", BOARD_ID)


@sio.event
def disconnect():
    print("[SERVER] Disconnected from server")


# UDP SOCKET - COMMUNICATION WITH PYNQ
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #why are we using DGRAM? what are the other options?
sock.bind(("0.0.0.0", PYNQ_PORT)) #which IP address and port program should listen on for incoming packets. 0.0.0.0 means listen on any network interface.

print("\n[UDP] Listening for shapes from PYNQ")
print("[UDP] Port:", PYNQ_PORT)


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


            # Check whether shape contains a type (has it been classified or not?)
            if "type" not in shape:
                print("[ERROR] Missing 'type' field")
                continue

            # Add object_id if missing
            if "object_id" not in shape:
                shape["object_id"] = "obj_" + str(random.randint(1000, 9999))
                print("[CLIENT] Generated object_id:", shape["object_id"])


            # Send to whiteboard server
            message = {
                "event": "ADD_OBJECT",
                "payload": shape
            }

            print("\n[CLIENT] Sending to server:")
            print(json.dumps(message, indent=2)) #Python Object -> JSON string

            sio.emit("board_event", message)

            print("[CLIENT] Shape forwarded successfully")

    except KeyboardInterrupt:

        print("\n[CLIENT] Stopping client...")

    finally:

        sock.close()
        sio.disconnect()

        print("[CLIENT] Shutdown complete")