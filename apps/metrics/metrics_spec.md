# Metrics Collected
## 1. Full Gesture Recognition Latency
Measures the total time required for gesture classification using the pynq.
*Definition: full_gesture_recognition_latency = shape_fully_classified − gesture_end*
Where:
gesture_end is in test.py. Defined by the moment the user finishes drawing a gesture.
shape_fully_classified is in test.py. Defined by the moment the FPGA reply is processed by the local laptop.
This latency includes:
- preprocessing on the laptop
- Ethernet transfer to the Pynq
- Pynq processing
- Ethernet transfer back to the laptop

Note: FPGA-only latency of the shape classification hardware module is measured separately by Aryan's hardware instrumentation. This will be a separate metric.

## 2. FPGA Latency
See above.

## 3. Network Latency
Measures the time required to transmit drawing updates from the laptop to the browser via the server. 
*Definition: network_latency = whiteboard_receive - sent_to_network*
Where:
send_to_network is in pynq_client.py. Defined by the moment the laptop sends the drawing update to the server.
whiteboard_receive is in whiteboard.html. Defined by the moment the browser receives the drawing update message.

## 4. End-to-End Latency
Measured the full latency from gesture completion to visible rendering on the shared whiteboard.
*Definition: end_to_end_latency = render_done - gesture_end*
Where:
render_done is in whiteboard.html. Defined by the moment the drawing becomes visible on the canvas.
This metric represents user-perceived responsiveness of the system.

# Event List
The following events are logged:

gesture_end
shape_fully_classified
send_to_network
whiteboard_receive
render_done

# Gesture message structure
Each gesture message is defined in test.py (unchanged):
    shape = {
        "object_id": object_id,
        "type": "stroke" if label in ("freehand", "line") else label,
        "created_at": timestamp_ms,
        "source": "capture_client",
    }

Here, timestamp_ms represents shape_fully_classified. This will be the reference for other calculations.

Note: I have changed where object_id is defined.


# Timestamp source
All Python code uses: time.perf_counter()
Browser code uses: performance.now()
Both provide monotonic high-res clocks suitable for latency measurement.
As all code is exectued on the same laptop, the time difference measurement is valid. 

# Logging Format
Events are written in JSON Lines format:
metrics/events.jsonl

Example entry:
{
    "object_id": "abc123",
    "event": "render_done"
    "timestamp": 1234.789
}

# Metrics Folder Architecture
metrics/
|
|-- metrics_spec.md
|-- logger.py
|-- analyse_metrics.py
|-- events.jsonl

Where:
metrics_spec.md: defines metric system
logger.py: shared event logger
analyse_metrics.py: compute metrics
events.jsonl: event log output