# websocket_manager.py
import asyncio
import json
import time
import websockets
from typing import Dict, Any
from queue import Queue
from threading import Lock

class WebSocketManager:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.connected_clients = set()
        self.message_queue = Queue()
        self.lock = Lock()
        
    async def start_server(self):
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=None
            )
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            # Démarrer la tâche de traitement des messages
            asyncio.create_task(self.process_message_queue())
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            
    async def process_message_queue(self):
        while True:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                await self._broadcast_message(message)
            await asyncio.sleep(0.01)  # Petit délai pour ne pas surcharger le CPU
            
    async def _broadcast_message(self, message):
        if not self.connected_clients:
            return
            
        disconnected = set()
        for websocket in self.connected_clients:
            try:
                await websocket.send(message)
                print(f"Message sent to client: {message}")
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.add(websocket)
        
        for websocket in disconnected:
            self.connected_clients.remove(websocket)
            
    def queue_message(self, face_data: Dict[str, Any]):
        message = json.dumps({
            "event_type": "face_data",
            "data": face_data,
            "timestamp": time.time()
        })
        self.message_queue.put(message)
        print(f"Message queued: {message}")

    async def handle_client(self, websocket):
        try:
            with self.lock:
                self.connected_clients.add(websocket)
            print(f"New client connected. Total clients: {len(self.connected_clients)}")
            
            async for _ in websocket:
                pass
                
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            with self.lock:
                self.connected_clients.remove(websocket)
            print(f"Client disconnected. Remaining clients: {len(self.connected_clients)}")