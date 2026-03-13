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


sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
app = web.Application() #create web server
sio.attach(app) #attach websockets to the server to allow continuous information exchange between server and client 

#in memory board state
boards = {} #dictionary storing board contents
DEFAULT_BOARD = "board_1"
boards[DEFAULT_BOARD] = {}

connected_clients = set() #tracks connected sockets
last_event_time = {}



#Client Lifecycle Events
@sio.event
async def connect(sid, environ):
    connected_clients.add(sid)
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    connected_clients.discard(sid)
    logger.info(f"Client disconnected: {sid}")


#Join Board
@sio.event
async def join_board(sid, data):
    board_id = data["board_id"]

    if board_id not in boards:
        boards[board_id] = {}

    await sio.save_session(sid, {"board_id": board_id})

    await sio.enter_room(sid, board_id)

    logger.info(f"{sid} joined {board_id}")

    #sends the full current board state back to that client with LOAD_BOARD
    await sio.emit("LOAD_BOARD", boards[board_id], to=sid) 

#Register PYNQ - separate to join_board
@sio.event
async def register_pynq(sid, data=None):
    await sio.save_session(sid, {
        "client_type":"pynq",
        "board_id":DEFAULT_BOARD
    })

    logger.info(f"{sid} registered for pynq {DEFAULT_BOARD}")


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
        shape_type = shape.get("type", {})
        params = shape.get("params",{})
        
        if(shape_type != "undo"): 
            boards[board_id][object_id] = shape
        else: 
            print("code for undo")
        
        await sio.emit("whiteboard_event", shape, room=board_id)
    
    except Exception: 
        logger.exception("Error receiving shape from pynq")

    

@sio.event 
async def whiteboard_event(sid, shape):
    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02: 
        return 
    else: 
        last_event_time[sid] = now
    
    try: 
        session = await sio.get_session(sid)
        board_id = session.get("board_id", DEFAULT_BOARD)

        if board_id not in boards: 
            boards[board_id] = {}
        
        
    
    except Exception: 
        logger.exception("Error with whiteboard event")


# Run Server
web.run_app(app, port=5000)