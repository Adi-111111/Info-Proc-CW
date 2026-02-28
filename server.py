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

def validate_message(message):
    if not isinstance(message, dict):
        return False, "Message must be a dictionary"

    if "event" not in message:
        return False, "Missing 'event' field"

    if "payload" not in message:
        return False, "Missing 'payload' field"

    if message["event"] not in ["ADD_OBJECT", "REMOVE_OBJECT"]:
        return False, "Invalid event type"

    payload = message["payload"]

    if not isinstance(payload, dict):
        return False, "Payload must be a dictionary"

    if message["event"] == "ADD_OBJECT":
        required = ["object_id", "type"]
        for field in required:
            if field not in payload:
                return False, f"Missing '{field}' in payload"

    if message["event"] == "REMOVE_OBJECT":
        if "object_id" not in payload:
            return False, "Missing 'object_id' in REMOVE_OBJECT"

    return True, None

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
app = web.Application()
sio.attach(app)

# In-memory board storage
# So far we are emulating multi-board capacity- in practice we always use "board1", can adjust the code for this if needed.
# May need to adjust some parts to use DEFAULT_BOARD rather than reading board_id from the client.
boards = {}
DEFAULT_BOARD = "board1"
boards[DEFAULT_BOARD] = {}

# Connected clients list
connected_clients = set()

# Adding basic rate limiting
last_event_time = {}

@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def join_board(sid, data):
    board_id = data["board_id"]
    if board_id not in boards:
        boards[board_id] = {}
    await sio.save_session(sid, {"board_id": board_id})
    logger.info(f"{sid} joined {board_id}")

    # Send existing board state
    await sio.emit("LOAD_BOARD", boards[board_id], to=sid)

@sio.event
async def board_event(sid, message):

    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02:
        # Ignore events faster than 50Hz
        return
    last_event_time[sid] = now

    try:
        valid, error = validate_message(message)
        if not valid:
            logger.warning(f"Invalid message from {sid}: {error}")
            return

        session = await sio.get_session(sid)
        board_id = session.get("board_id")

        if board_id not in boards:
            boards[board_id] = {}

        # Adding maximum object count protection
        if len(boards[board_id]) > MAX_OBJECTS:
            logger.warning("Board object limit reached")
            return

        event = message["event"]
        payload = message["payload"]

        if event == "ADD_OBJECT":
            boards[board_id][payload["object_id"]] = payload

        elif event == "REMOVE_OBJECT":
            boards[board_id].pop(payload["object_id"], None)

        await sio.emit("board_event", message)

    except Exception:
        logger.exception("Error handling board_event")

web.run_app(app, port=5000)