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
DEFAULT_BOARD = "board1"
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




#Board Event - Handles normal whiteboard updates from the frontend
#Expects a message with an event and payload
#Defined asynchronously 
@sio.event
async def board_event(sid, message):

    #checks the time of the most recent event sent from this client - if too soon then return otherwise set the last_event_time as now. (rate_limiting)
    now = time.time()
    if sid in last_event_time and now - last_event_time[sid] < 0.02:
        return
    last_event_time[sid] = now 

    try:
        session = await sio.get_session(sid) #get the stored session information for the client indentified by this sid 
        board_id = session.get("board_id") #get the board to apply the event to? currently hardcoded, how do we change this?

        event = message["event"]
        payload = message["payload"]

        #Stores the object in the board directory
        if event == "ADD_OBJECT":
            boards[board_id][payload["object_id"]] = payload

        #Removes the object by object_id
        elif event == "REMOVE_OBJECT":
            boards[board_id].pop(payload["object_id"], None)

        #Broadcast event to everyone in the board room
        await sio.emit("board_event", message, room=board_id)

    except Exception:
        logger.exception("Error handling board_event")




# #Shape Event - Handles shapes coming from PYNQ bridge
# #Expects data with id, type and params 
# #Supports polyline and rectangle
# #Converts it into the same board object format and broadcasts it as an ADD_OBJECT
# @sio.event
# async def shape_event(sid, data):

#     now = time.time()
#     if sid in last_event_time and now - last_event_time[sid] < 0.02:
#         return
#     last_event_time[sid] = now

#     try:

#         obj_id = data.get("id")
#         obj_type = data.get("type")
#         params = data.get("params", {})

#         if not obj_id or not obj_type:
#             logger.warning("Invalid shape_event")
#             return

#         session = await sio.get_session(sid)
#         board_id = session.get("board_id")

#         payload = {
#             "object_id": obj_id,
#             "type": obj_type
#         }

#         if obj_type == "polyline":
#             payload["points"] = params.get("points", [])

#         elif obj_type == "rectangle":
#             payload["corners"] = params.get("corners", [])

#         else:
#             logger.warning(f"Unknown shape type: {obj_type}")
#             return

#         boards[board_id][obj_id] = payload

#         message = {
#             "event": "ADD_OBJECT",
#             "payload": payload
#         }

#         #Broadcast Board Event to Room
#         await sio.emit(
#             "board_event",
#             message,
#             room=board_id
#         )

#         logger.info(f"shape_event → broadcast {obj_type}")

#     except Exception:
#         logger.exception("Error handling shape_event")




# Run Server
web.run_app(app, port=5000)