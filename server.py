import socketio
from aiohttp import web
import logging
import time

MAX_OBJECTS = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
app = web.Application()
sio.attach(app)

boards = {}
DEFAULT_BOARD = "board1"
boards[DEFAULT_BOARD] = {}

connected_clients = set()
last_event_time = {}

# =========================
# CLIENT CONNECT
# =========================

@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    logger.info(f"Client disconnected: {sid}")

# =========================
# JOIN BOARD
# =========================

@sio.event
async def join_board(sid, data):

    board_id = data["board_id"]

    if board_id not in boards:
        boards[board_id] = {}

    await sio.save_session(sid, {"board_id": board_id})

    # 🔴 THIS WAS MISSING
    await sio.enter_room(sid, board_id)

    logger.info(f"{sid} joined {board_id}")

    await sio.emit("LOAD_BOARD", boards[board_id], to=sid)

# =========================
# NORMAL BOARD EVENTS
# =========================

@sio.event
async def board_event(sid, message):

    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02:
        return
    last_event_time[sid] = now

    try:

        session = await sio.get_session(sid)
        board_id = session.get("board_id")

        event = message["event"]
        payload = message["payload"]

        if event == "ADD_OBJECT":
            boards[board_id][payload["object_id"]] = payload

        elif event == "REMOVE_OBJECT":
            boards[board_id].pop(payload["object_id"], None)

        # 🔴 BROADCAST ONLY TO BOARD ROOM
        await sio.emit(
            "board_event",
            message,
            room=board_id
        )

    except Exception:
        logger.exception("Error handling board_event")

# =========================
# SHAPE EVENT (FROM PYNQ)
# =========================

@sio.event
async def shape_event(sid, data):

    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02:
        return
    last_event_time[sid] = now

    try:

        obj_id = data.get("id")
        obj_type = data.get("type")
        params = data.get("params", {})

        if not obj_id or not obj_type:
            logger.warning("Invalid shape_event")
            return

        session = await sio.get_session(sid)
        board_id = session.get("board_id")

        payload = {
            "object_id": obj_id,
            "type": obj_type
        }

        if obj_type == "polyline":
            payload["points"] = params.get("points", [])

        elif obj_type == "rectangle":
            payload["corners"] = params.get("corners", [])

        else:
            logger.warning(f"Unknown shape type: {obj_type}")
            return

        boards[board_id][obj_id] = payload

        message = {
            "event": "ADD_OBJECT",
            "payload": payload
        }

        # 🔴 BROADCAST TO BOARD ROOM
        await sio.emit(
            "board_event",
            message,
            room=board_id
        )

        logger.info(f"shape_event → broadcast {obj_type}")

    except Exception:
        logger.exception("Error handling shape_event")

# =========================
# RUN SERVER
# =========================

web.run_app(app, port=5000)