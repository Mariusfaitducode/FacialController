# FacialController

Real-time face tracking that turns eye blinks and mouth opening into control events, streamed over WebSocket so any app or game can use your face as a controller.

## What it does
- Captures the webcam and tracks up to 4 faces at once with MediaPipe Face Mesh.
- Detects blinks from an eye aspect ratio computed on the face landmarks, and counts them per face.
- Detects whether the mouth is open from the inner-lip landmarks.
- Runs a WebSocket server (`ws://localhost:8765`) and pushes a JSON event to all connected clients whenever a face's blink or mouth state changes.
- Answers `request_snapshot` messages from clients by sending a cropped JPEG of the requested face as a binary message.
- Shows a preview window with every detected face and its landmarks.

## Tech stack
Python, OpenCV, MediaPipe Face Mesh, `websockets`, asyncio.

## Run it locally
There is no requirements file; install the libraries used by the code:

```bash
pip install opencv-python mediapipe websockets numpy
cd src
python main.py
```
Then connect a WebSocket client to `ws://localhost:8765` (a small test client is in `src/websocket/test_websocket_client.py`). Press `q` or `Esc` in the preview window to quit.

Example event sent to clients:
```json
{"event_type": "face_data", "data": {"face_id": 0, "blink_detected": true, "mouth_open": false, "mouth_ratio": 0.05, "total_blinks": 3}, "timestamp": 1735000000.0}
```

## Context
Built in December 2024 to explore faces as an alternative input for games and interactive applications. `mesh_map.jpg` and `mediapipe_face_landmark_fullsize.png` are MediaPipe landmark reference maps used to pick the eye and mouth landmark indices.
