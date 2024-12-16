# websocket_manager.py
import asyncio
import json
import time
import websockets
from typing import Dict, Any
from queue import Queue
from threading import Lock
from .messages import MessageType, BinaryMessageType, WebSocketMessage
import struct
import cv2
import json

class WebSocketManager:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.connected_clients = set()
        self.message_queue = Queue()
        self.lock = Lock()
        self.connections = set()
        self.face_trackers = {} 
        
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
                if isinstance(message, bytes):
                    await self._broadcast_binary(message)
                else:
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
            "event_type": MessageType.FACE_DATA.value,
            "data": face_data,
            "timestamp": time.time()
        })
        self.message_queue.put(message)
        print(f"Message queued: {message}")


    def queue_snapshot(self, face_id: int, frame):
        """Prépare et met en file d'attente un snapshot en format binaire"""
        try:
            # Compression JPEG de l'image
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = img_encoded.tobytes()
            
            # Structure du message binaire:
            # [1 byte: message_type][4 bytes: face_id][4 bytes: data_length][N bytes: image_data]
            header = struct.pack('!BII', 
                BinaryMessageType.SNAPSHOT,  # 1 byte pour le type
                face_id,                     # 4 bytes pour l'ID
                len(img_bytes)               # 4 bytes pour la taille
            )
            
            binary_message = header + img_bytes
            self.message_queue.put(binary_message)
            print(f"Binary snapshot queued for face_id: {face_id}, size: {len(binary_message)} bytes")
            
        except Exception as e:
            print(f"Error preparing binary snapshot: {e}")

    def queue_face_data(self, face_data: dict):
        """Met en file d'attente les données de visage en JSON"""
        try:
            json_data = json.dumps(face_data).encode('utf-8')
            
            # Structure similaire mais pour les données JSON
            header = struct.pack('!BII',
                BinaryMessageType.FACE_DATA,
                face_data['face_id'],
                len(json_data)
            )
            
            binary_message = header + json_data
            self.message_queue.put(binary_message)
            
        except Exception as e:
            print(f"Error preparing face data: {e}")

    async def handle_client(self, websocket):
        try:
            print("New client connected")
            self.connected_clients.add(websocket)
            self.connections.add(websocket)
            
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == MessageType.REQUEST_SNAPSHOT.value:
                    print(f"Snapshot requested for face_id: {data.get('face_id')}")
                    face_id = data.get("face_id")
                    if face_id in self.face_trackers:
                        self.face_trackers[face_id].request_snapshot()
                        
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
            pass
        finally:
            self.connected_clients.remove(websocket)
            self.connections.remove(websocket)


    def register_face_tracker(self, face_id, tracker):
        """Enregistre un face tracker"""
        self.face_trackers[face_id] = tracker
        print(f"Face tracker registered for face_id: {face_id}")


    def unregister_face_tracker(self, face_id):
        """Désenregistre un face tracker"""
        if face_id in self.face_trackers:
            del self.face_trackers[face_id]
            print(f"Face tracker unregistered for face_id: {face_id}")

    async def _broadcast_binary(self, binary_data):
        """Envoie des données binaires à tous les clients connectés"""
        if not self.connected_clients:
            return
            
        disconnected = set()
        for websocket in self.connected_clients:
            try:
                await websocket.send(binary_data)
            except Exception as e:
                print(f"Error sending binary data: {e}")
                disconnected.add(websocket)
        
        for websocket in disconnected:
            self.connected_clients.remove(websocket)