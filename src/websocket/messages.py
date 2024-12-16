from enum import Enum
import json
import base64
import cv2


from enum import IntEnum

class BinaryMessageType(IntEnum):
    FACE_DATA = 1
    SNAPSHOT = 2

class MessageType(Enum):
    # Face data : blink_count, mouth_open events
    FACE_DATA = "face_data"
    # Snapshot data : image of a face
    REQUEST_SNAPSHOT = "request_snapshot"
    SNAPSHOT_DATA = "snapshot_data"

class WebSocketMessage:
    @staticmethod
    def create_face_data(face_id: int, blink_count: int, mouth_open: bool):
        return json.dumps({
            "type": MessageType.FACE_DATA.value,
            "face_id": face_id,
            "blink_count": blink_count,
            "mouth_open": mouth_open
        })

    @staticmethod
    def create_snapshot_response(face_id: int, image_data):
        # Convertir l'image en base64
        _, buffer = cv2.imencode('.jpg', image_data)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return json.dumps({
            "type": MessageType.SNAPSHOT_DATA.value,
            "face_id": face_id,
            "image": img_base64
        }) 