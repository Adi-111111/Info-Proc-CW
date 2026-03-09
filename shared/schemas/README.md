# Board Event Schemas

This document describes all event types used across the system, including Socket.IO events between clients and server, and UDP messages between the capture client and PYNQ bridge.

---

## Socket.IO Events

### `join_board`
**Direction**: Client → Server

Sent by a client to join a collaborative board session.

```json
{
  "board_id": "board1"
}
```

---

### `LOAD_BOARD`
**Direction**: Server → Client

Sent by the server immediately after a client joins. Contains the full current board state as a dictionary of objects keyed by `object_id`.

```json
{
  "obj_id_1": {
    "object_id": "obj_id_1",
    "type": "stroke",
    "points": [[100, 200], [105, 205], [110, 210]]
  },
  "obj_id_2": {
    "object_id": "obj_id_2",
    "type": "rectangle",
    "corners": [[100.0, 150.0], [200.0, 150.0], [200.0, 250.0], [100.0, 250.0]]
  },
  "obj_id_3": {
    "object_id": "obj_id_3",
    "type": "circle",
    "cx": 150.5,
    "cy": 200.25,
    "r": 50.75
  }
}
```

---

### `board_event`
**Direction**: Client ↔ Server ↔ All clients on board

The primary event type for all board mutations. Wraps an inner event type and payload. The server broadcasts to all other clients on the same board.

#### `ADD_OBJECT` — Stroke
```json
{
  "event": "ADD_OBJECT",
  "payload": {
    "object_id": "obj_12345",
    "type": "stroke",
    "points": [[100, 200], [105, 205], [110, 210]]
  }
}
```

#### `ADD_OBJECT` — Rectangle
```json
{
  "event": "ADD_OBJECT",
  "payload": {
    "object_id": "obj_12345",
    "type": "rectangle",
    "corners": [[100.0, 150.0], [200.0, 150.0], [200.0, 250.0], [100.0, 250.0]]
  }
}
```

#### `ADD_OBJECT` — Circle
```json
{
  "event": "ADD_OBJECT",
  "payload": {
    "object_id": "obj_12345",
    "type": "circle",
    "cx": 150.5,
    "cy": 200.25,
    "r": 50.75
  }
}
```

#### `REMOVE_OBJECT`
```json
{
  "event": "REMOVE_OBJECT",
  "payload": {
    "object_id": "obj_12345"
  }
}
```

---

### `shape_event`
**Direction**: PYNQ Bridge → Server

Sent by the PYNQ bridge client after the PYNQ board has recognised a shape from a raw stroke. The server converts this into a `board_event` / `ADD_OBJECT` and broadcasts it.

#### Polyline
```json
{
  "id": "obj_12345",
  "type": "polyline",
  "params": {
    "points": [[100, 200], [105, 205], [110, 210]]
  }
}
```

#### Rectangle
```json
{
  "id": "obj_12345",
  "type": "rectangle",
  "params": {
    "corners": [[100.0, 150.0], [200.0, 150.0], [200.0, 250.0], [100.0, 250.0]]
  }
}
```

#### Circle
```json
{
  "id": "obj_12345",
  "type": "circle",
  "params": {
    "cx": 150.5,
    "cy": 200.25,
    "r": 50.75
  }
}
```

---

## UDP Messages

### Capture Client → PYNQ (port 5005)

Raw stroke points sent for hardware shape recognition.

```json
{
  "stroke": [[100, 200], [105, 205], [110, 210]]
}
```

### PYNQ → PC Bridge (port 5006)

Recognised shape sent back to the PC after hardware processing.

#### Circle
```json
{
  "type": "circle",
  "cx": 150.5,
  "cy": 200.25,
  "r": 50.75
}
```

#### Rectangle
```json
{
  "type": "rectangle",
  "corners": [[100.0, 150.0], [200.0, 150.0], [200.0, 250.0], [100.0, 250.0]]
}
```

#### Stroke (fallback — no shape recognised)
```json
{
  "type": "stroke",
  "points": [[100, 200], [105, 205], [110, 210]]
}
```

---

## Object Types Summary

| Type | Fields |
|---|---|
| `stroke` | `points: [[x, y], ...]` |
| `polyline` | `points: [[x, y], ...]` |
| `rectangle` | `corners: [[x, y], [x, y], [x, y], [x, y]]` (top-left → clockwise) |
| `circle` | `cx: float`, `cy: float`, `r: float` |
