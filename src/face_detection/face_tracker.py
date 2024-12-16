import datetime
import cv2
import asyncio
from typing import Optional


from face_detection.blink_detection import extract_eye_region, analyze_eye_state, blinkRatio

from face_detection.mouth_detection import mouthRatio


# landmarks from mesh_map.jpg
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

CLOSED_EYE_THRESH = 3.3   # Seuil au-dessus duquel l'œil est considéré comme fermé
MIN_FRAMES = 5    

COLORS = {
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0),
    'RED': (0, 0, 255)
}

BLINK_COUNTER = 0
TOTAL_BLINKS = 0


FONT = cv2.FONT_HERSHEY_SIMPLEX



# Définissons des points plus précis pour la bouche
INNER_LIPS = {
    'top': [13],           # Point central supérieur
    'bottom': [14],        # Point central inférieur
    'left': [78],          # Point gauche
    'right': [308]         # Point droit
}

def landmarksDetection(frame, frame_show, landmarks, draw=False):
    image_height, image_width = frame.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) 
                       for point in landmarks.landmark]
    
    if draw:
        for idx, point in enumerate(mesh_coordinates):
            if idx in LEFT_EYE or idx in RIGHT_EYE:
                # Points des yeux en bleu
                cv2.circle(frame_show, point, 2, COLORS['BLUE'], -1)
            elif idx in LIPS:
                # Points de la bouche en rouge
                cv2.circle(frame_show, point, 2, COLORS['RED'], -1)
            else:
                # Autres points en vert
                cv2.circle(frame_show, point, 2, COLORS['GREEN'], -1)
                
    return mesh_coordinates


# Définir la variable globale
websocket_loop = None

# Créer une classe pour gérer l'état de chaque visage
class FaceTracker:
    def __init__(self, face_id, websocket_manager=None):
        global websocket_loop  # Accéder à la variable globale
        self.face_id = face_id
        self.blink_counter = 0
        self.total_blinks = 0
        self.previous_eye = None
        self.window_name = f'Face {face_id}'
        self.websocket_manager = websocket_manager
        self.last_blink_state = False
        self.last_mouth_state = False
        self.snapshot_requested = False  # Nouveau flag
        
    def emit_face_state(self, is_blinking: bool, mouth_ratio: float):
        if self.websocket_manager:
            face_data = {
                "face_id": self.face_id,
                "blink_detected": is_blinking,
                "mouth_open": mouth_ratio > 0.2,
                "mouth_ratio": mouth_ratio,
                "total_blinks": self.total_blinks
            }
            try:
                print(f"Queueing data: {face_data}")
                self.websocket_manager.queue_message(face_data)
                print("Data queued successfully")
            except Exception as e:
                print(f"Error in emit_face_state: {e}")
    
    def update(self, frame, frame_show, landmarks):
        # Créer une copie du frame pour ce visage
        face_frame = frame.copy()
        
        # Dessiner les landmarks pour ce visage
        mesh_coordinates = landmarksDetection(face_frame, frame_show, landmarks, True)
        
        # Calculer le ratio des yeux
        total_ratio, left_ratio, right_ratio = blinkRatio(face_frame, mesh_coordinates, RIGHT_EYE, LEFT_EYE)
        
        # Extraire et analyser la région des yeux
        eye_region, eye_bbox = extract_eye_region(face_frame, mesh_coordinates, LEFT_EYE + RIGHT_EYE)
        
        eye_change = 0
        eye_area = 0
        if self.previous_eye is not None and eye_region.size > 0:
            if eye_region.shape == self.previous_eye.shape:
                eye_change, eye_area = analyze_eye_state(eye_region, self.previous_eye)
        
        self.previous_eye = eye_region.copy() if eye_region.size > 0 else None
        
        # Détecter les clignements
        is_blinking = total_ratio > CLOSED_EYE_THRESH
        if is_blinking:
            self.blink_counter += 1
        else:
            if self.blink_counter > MIN_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0
        
        # Afficher les informations
        cv2.rectangle(face_frame, (20, 120), (350, 300), (0,0,0), -1)
        cv2.putText(face_frame, f'Blink Counter: {self.blink_counter}', (30, 150), FONT, 0.7, COLORS['RED'], 2)
        cv2.putText(face_frame, f'Total Blinks: {self.total_blinks}', (30, 180), FONT, 0.7, COLORS['RED'], 2)
        cv2.putText(face_frame, f'Eye Ratio: {total_ratio:.2f}', (30, 210), FONT, 0.7, COLORS['RED'], 2)
        cv2.putText(face_frame, f'Change: {eye_change:.2f} Area: {eye_area:.2f}', (30, 240), FONT, 0.7, COLORS['RED'], 2)

        #####################################################
        # * Mouth Detection
        #####################################################

        # Ajout de la détection de l'ouverture de la bouche
        mouth_ratio = mouthRatio(mesh_coordinates, INNER_LIPS)
        
        # Émettre l'état si changement
        if (is_blinking != self.last_blink_state or (mouth_ratio > 0.2) != self.last_mouth_state):
            self.emit_face_state(is_blinking, mouth_ratio)
            self.last_blink_state = is_blinking
            self.last_mouth_state = mouth_ratio > 0.2
        
        # Seuil pour déterminer si la bouche est ouverte
        if mouth_ratio > 0.2:
            cv2.putText(face_frame, "Bouche ouverte: " + str(mouth_ratio), (30, 700), FONT, 1, COLORS['RED'], 2)
        else:
            cv2.putText(face_frame, "Bouche fermee: " + str(mouth_ratio), (30, 700), FONT, 1, COLORS['GREEN'], 2)
        
        # Afficher la fenêtre pour ce visage
        # cv2.imshow(self.window_name, face_frame)

        # Après le traitement du visage
        # processed_face = frame[y:y+h, x:x+w]
        processed_face = frame
        
        # Si un snapshot a été demandé, l'envoyer via websocket
        if self.snapshot_requested:
            self.websocket_manager.queue_snapshot(self.face_id, processed_face)
            self.snapshot_requested = False  # Réinitialiser le flag
            
        return frame_show

    def request_snapshot(self):
        """Méthode pour demander un snapshot lors de la prochaine update"""
        self.snapshot_requested = True


        
