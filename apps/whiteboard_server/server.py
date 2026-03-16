import socketio
from aiohttp import web
import logging
import time
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from decimal import Decimal
import json
from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parents[1]   # apps/
#METRICS_DIR = BASE_DIR / "metrics"
#METRICS_FILE = METRICS_DIR / "events.jsonl"

METRICS_FILE = Path(__file__).resolve().parent / "events.jsonl"

def log_metric(entry: dict):
    #METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

MAX_OBJECTS = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# DYNAMODB SETUP
# =========================

TABLE_NAME = "whiteboard_objects"
DEFAULT_BOARD = "board1"
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-2")

logger.info(f"Connecting to AWS DynamoDB in region {AWS_REGION} ...")

dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
    config=Config(connect_timeout=5, read_timeout=5, retries={"max_attempts": 3}),
)

# Quick connectivity check — list tables to verify DynamoDB is reachable
try:
    dynamodb_client = dynamodb.meta.client
    existing_tables = dynamodb_client.list_tables()["TableNames"]
    logger.info(f"✅ Connected to AWS DynamoDB ({AWS_REGION}) — existing tables: {existing_tables}")
except Exception as e:
    logger.error(f"❌ Cannot reach AWS DynamoDB ({AWS_REGION}): {e}")
    raise SystemExit(1)

def _create_table_if_not_exists():
    """Create the DynamoDB table if it doesn't already exist."""
    try:
        t = dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[
                {"AttributeName": "board_id", "KeyType": "HASH"},
                {"AttributeName": "object_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "board_id", "AttributeType": "S"},
                {"AttributeName": "object_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        t.wait_until_exists()
        logger.info(f"Created DynamoDB table '{TABLE_NAME}'")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            logger.info(f"DynamoDB table '{TABLE_NAME}' already exists")
        else:
            raise

_create_table_if_not_exists()
table = dynamodb.Table(TABLE_NAME)

# Ensure the default board exists
table.put_item(Item={"board_id": DEFAULT_BOARD, "object_id": "__meta__", "name": "Default Board"})

# =========================
# DYNAMO HELPERS
# =========================

def _float_to_decimal(obj):
    """Recursively convert floats to Decimals for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_float_to_decimal(i) for i in obj]
    return obj


def _decimal_to_float(obj):
    """Recursively convert Decimals back to floats/ints."""
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj


def db_list_boards() -> list:
    """List all boards by scanning for __meta__ items."""
    response = table.scan(
        FilterExpression=boto3.dynamodb.conditions.Attr("object_id").eq("__meta__")
    )
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("object_id").eq("__meta__"),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))
    return [{"board_id": item["board_id"], "name": item.get("name", item["board_id"])} for item in items]


def db_create_board(board_id: str, name: str):
    """Register a new board by writing a __meta__ item."""
    table.put_item(Item={"board_id": board_id, "object_id": "__meta__", "name": name})


def db_put_object(board_id: str, payload: dict):
    """Write a board object to DynamoDB."""
    item = {
        "board_id": board_id,
        "object_id": payload["object_id"],
        "data": _float_to_decimal(payload),
    }
    table.put_item(Item=item)


def db_delete_object(board_id: str, object_id: str):
    """Delete a board object from DynamoDB."""
    table.delete_item(Key={"board_id": board_id, "object_id": object_id})


def db_delete_board(board_id: str):
    """Delete all items for a board (including __meta__) from DynamoDB."""
    expr_names = {"#bid": "board_id", "#oid": "object_id"}
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("board_id").eq(board_id),
        ProjectionExpression="#bid, #oid",
        ExpressionAttributeNames=expr_names,
    )
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("board_id").eq(board_id),
            ProjectionExpression="#bid, #oid",
            ExpressionAttributeNames=expr_names,
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))
    with table.batch_writer() as batch:
        for item in items:
            batch.delete_item(Key={"board_id": item["board_id"], "object_id": item["object_id"]})


def db_load_board(board_id: str) -> dict:
    """Load all objects for a board from DynamoDB. Returns {object_id: payload}."""
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("board_id").eq(board_id)
    )
    result = {}
    for item in response.get("Items", []):
        if item["object_id"] == "__meta__":
            continue
        data = _decimal_to_float(item["data"])
        result[data["object_id"]] = data

    while "LastEvaluatedKey" in response:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("board_id").eq(board_id),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        for item in response.get("Items", []):
            if item["object_id"] == "__meta__":
                continue
            data = _decimal_to_float(item["data"])
            result[data["object_id"]] = data

    return result


# =========================
# SOCKET.IO + APP SETUP
# =========================

sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
app = web.Application()
sio.attach(app)

# In-memory cache (loaded from DynamoDB on first access)
boards = {}
boards[DEFAULT_BOARD] = db_load_board(DEFAULT_BOARD)

# Insertion-ordered list of object_ids per board — used for undo
board_order = {}
board_order[DEFAULT_BOARD] = list(boards[DEFAULT_BOARD].keys())

# Redo stack: stores full payloads of undo'd objects (already deleted from DB)
redo_stacks = {}
redo_stacks[DEFAULT_BOARD] = []

# Supported object types for board_event ADD_OBJECT:
#   circle:    {"object_id": "...", "type": "circle", "cx": 150.5, "cy": 200.25, "r": 50.75}
#   rectangle: {"object_id": "...", "type": "rectangle", "corners": [[x,y],[x,y],[x,y],[x,y]]}
#   triangle:  {"object_id": "...", "type": "triangle", "corners": [[x1,y1],[x2,y2],[x3,y3]]}
#   stroke:    {"object_id": "...", "type": "stroke", "points": [[x,y], ...]}

connected_clients = set()
pynq_clients = set()
last_event_time = {}
active_pynq_board = DEFAULT_BOARD

# =========================
# CLIENT LIFECYCLE
# =========================

@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    pynq_clients.discard(sid)
    logger.info(f"Client disconnected: {sid}")

# =========================
# JOIN BOARD
# =========================

@sio.event
async def join_board(sid, data):
    board_id = data["board_id"]

    if board_id not in boards:
        boards[board_id] = db_load_board(board_id)
        board_order[board_id] = list(boards[board_id].keys())
        redo_stacks[board_id] = []

    await sio.save_session(sid, {"board_id": board_id})
    await sio.enter_room(sid, board_id)
    logger.info(f"{sid} joined {board_id}")

    # Send full current board state to this client
    await sio.emit("LOAD_BOARD", boards[board_id], to=sid)

# =========================
# LIST BOARDS
# =========================

@sio.event
async def list_boards(sid):
    board_list = db_list_boards()
    await sio.emit("LIST_BOARDS", board_list, to=sid)

# =========================
# CREATE BOARD
# =========================

@sio.event
async def create_board(_sid, data):
    board_id = data["board_id"]
    name = data["name"]
    db_create_board(board_id, name)
    boards[board_id] = {}
    await sio.emit("BOARD_CREATED", {"board_id": board_id, "name": name})

# =========================
# DELETE BOARD
# =========================

@sio.event
async def delete_board(_sid, data):
    try:
        board_id = data["board_id"]
        db_delete_board(board_id)
        boards.pop(board_id, None)
        await sio.emit("BOARD_DELETED", {"board_id": board_id})
    except Exception:
        logger.exception("Error deleting board")

# =========================
# PYNQ EVENTS
# =========================

@sio.event
async def register_pynq(sid, data=None):
    global active_pynq_board
    pynq_clients.add(sid)
    board_id = (data or {}).get("board_id", active_pynq_board)
    if board_id not in boards:
        boards[board_id] = db_load_board(board_id)
        board_order[board_id] = list(boards[board_id].keys())
        redo_stacks[board_id] = []
    active_pynq_board = board_id
    await sio.save_session(sid, {
        "client_type": "pynq",
        "board_id": board_id,
    })
    logger.info(f"{sid} registered as pynq on {board_id}")


@sio.event
async def set_pynq_board(_sid, data):
    global active_pynq_board
    board_id = data["board_id"]

    if board_id not in boards:
        boards[board_id] = db_load_board(board_id)
        board_order[board_id] = list(boards[board_id].keys())
        redo_stacks[board_id] = []

    active_pynq_board = board_id

    for pynq_sid in list(pynq_clients):
        try:
            session = await sio.get_session(pynq_sid)
        except KeyError:
            pynq_clients.discard(pynq_sid)
            continue

        session["client_type"] = "pynq"
        session["board_id"] = board_id
        await sio.save_session(pynq_sid, session)

    logger.info(f"Active PYNQ board set to {board_id}")


@sio.event
async def pynq_event(sid, shape): 
    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02: 
        return
    else: 
        last_event_time[sid] = now
    
    try: 
        session = await sio.get_session(sid)
        board_id = session.get("board_id", DEFAULT_BOARD)

        object_id = shape.get("object_id", {})
        if not object_id:
            logger.warning("Received shape without object_id")
            return

        boards.setdefault(board_id, {})
        board_order.setdefault(board_id, [])
        redo_stacks.setdefault(board_id, [])
        boards[board_id][object_id] = shape
        board_order[board_id].append(object_id)
        db_put_object(board_id, shape)

        log_metric({  # metrics
            "event": "whiteboard_receive",
            "timestamp": time.time(),
            "board_id": board_id,
            "object_id": object_id,
            "shape_type": shape.get("type"),
        })

        await sio.emit("whiteboard_event", shape, room=board_id)

        log_metric({  # metrics
            "event": "render_done", 
            "timestamp": time.time(),
            "board_id": board_id,
            "object_id": object_id,
        })
    
    except Exception: 
        logger.exception("Error receiving shape from pynq")


# =========================
# UNDO LAST OBJECT
# =========================

@sio.event
async def undo_last(_sid, data):
    try:
        board_id = data.get("board_id", DEFAULT_BOARD)

        order = board_order.get(board_id, [])
        if not order:
            logger.info(f"undo_last: nothing to undo on {board_id}")
            return

        object_id = order.pop()

        payload = boards[board_id].pop(object_id, None)
        db_delete_object(board_id, object_id)

        if payload:
            redo_stacks.setdefault(board_id, []).append(payload)

        await sio.emit(
            "remove_object",
            {"object_id": object_id},
            room=board_id,
        )

        logger.info(f"undo_last: removed {object_id} from {board_id}, redo stack size={len(redo_stacks.get(board_id, []))}")

    except Exception:
        logger.exception("Error handling undo_last")


# =========================
# REDO LAST OBJECT
# =========================

@sio.event
async def redo_last(_sid, data):
    try:
        board_id = data.get("board_id", DEFAULT_BOARD)

        stack = redo_stacks.get(board_id, [])
        if not stack:
            logger.info(f"redo_last: nothing to redo on {board_id}")
            return

        payload = stack.pop()
        object_id = payload["object_id"]

        boards[board_id][object_id] = payload
        board_order.setdefault(board_id, []).append(object_id)
        db_put_object(board_id, payload)

        await sio.emit("whiteboard_event", payload, room=board_id)

        logger.info(f"redo_last: restored {object_id} to {board_id}, redo stack size={len(stack)}")

    except Exception:
        logger.exception("Error handling redo_last")

# RUN SERVER

web.run_app(app, port=5000)
