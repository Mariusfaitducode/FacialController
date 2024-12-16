import cv2
import base64
import json
import websockets
import asyncio

async def send_image(websocket):
    # Capture de l'image via OpenCV
    cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir l'image en JPG
        _, buffer = cv2.imencode('.jpg', frame)

        # Encoder l'image en Base64 pour l'envoyer
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Préparer un message JSON contenant l'image
        message = json.dumps({"type": "image", "data": image_base64})

        # Envoyer via WebSocket
        await websocket.send(message)

        await asyncio.sleep(0.033)  # Environ 30 FPS

    cap.release()

async def main():
    async with websockets.connect("ws://localhost:8765") as websocket:
        await send_image(websocket)

asyncio.run(main())
