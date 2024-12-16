import cv2
import mediapipe
import asyncio

from websocket.websocket_manager import WebSocketManager
from face_detection.face_tracker import *
import threading
# from concurrent.futures import ThreadPoolExecutor


mediapipe_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mediapipe_face_mesh.FaceMesh(
    max_num_faces=4,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

video_capture = cv2.VideoCapture(0)


# Dictionnaire pour suivre tous les visages
face_trackers = {}

# Création d'une nouvelle boucle d'événements pour le WebSocket
websocket_loop = None
websocket_manager = None

async def run_websocket_server():
    global websocket_manager
    websocket_manager = WebSocketManager()
    await websocket_manager.start_server()
    while True:
        await asyncio.sleep(0.1)

def start_websocket_server():
    global websocket_loop
    websocket_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(websocket_loop)
    websocket_loop.run_until_complete(run_websocket_server())

def main():
    # Démarrer le serveur WebSocket dans un thread séparé
    websocket_thread = threading.Thread(target=start_websocket_server)
    websocket_thread.daemon = True
    websocket_thread.start()

    # Attendre un peu que le serveur WebSocket démarre
    import time
    time.sleep(2)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        frame_show = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        
        # Nettoyer les fenêtres des visages qui ne sont plus détectés
        if not results.multi_face_landmarks:
            face_trackers.clear()
        else:
            current_faces = set()
            
            # Mettre à jour chaque visage détecté
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                current_faces.add(face_idx)
                
                # Créer ou obtenir le tracker pour ce visage
                if face_idx not in face_trackers:
                    face_trackers[face_idx] = FaceTracker(face_idx, websocket_manager)
                
                # Mettre à jour le tracker
                frame_show = face_trackers[face_idx].update(frame, frame_show, face_landmarks)
                # cv2.imshow(face_trackers[face_idx].window_name, face_frame)
            
            # Supprimer les trackers des visages qui ne sont plus détectés
            faces_to_remove = set(face_trackers.keys()) - current_faces
            for face_idx in faces_to_remove:
                # cv2.destroyWindow(face_trackers[face_idx].window_name)
                del face_trackers[face_idx]
        
        # Afficher le frame principal avec tous les visages détectés
        cv2.imshow('All Faces', frame_show)
        
        key = cv2.waitKey(2)
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()
    if websocket_loop:
        websocket_loop.stop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Programme arrêté par l'utilisateur")
    finally:
        cv2.destroyAllWindows()
        video_capture.release()
