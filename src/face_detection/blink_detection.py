import cv2
import numpy as np

from face_detection.utils import euclaideanDistance



# Blinking Ratio
def blinkRatio(image, landmarks, right_indices, left_indices):

    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    # Finding Distance Right Eye
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    # Finding Distance Left Eye
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    if rhDistance == 0 or rvDistance == 0 or lhDistance == 0 or lvDistance == 0:
        return 0, 0, 0

    # Finding ratio of LEFT and Right Eyes
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2
    return ratio, reRatio, leRatio




def extract_eye_region(frame, landmarks, eye_points):
    # Obtenir les coordonnées min et max pour créer un rectangle autour de l'œil
    x_coords = [landmarks[point][0] for point in eye_points]
    y_coords = [landmarks[point][1] for point in eye_points]
    
    # Ajouter une marge autour de l'œil
    margin = 5
    x_min, x_max = int(min(x_coords) - margin), int(max(x_coords) + margin)
    y_min, y_max = int(min(y_coords) - margin), int(max(y_coords) + margin)
    
    # Extraire la région de l'œil
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    # Convertir en niveaux de gris
    if eye_region.size > 0:
        eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        # Améliorer le contraste
        eye_region = cv2.equalizeHist(eye_region)
    
    return eye_region, (x_min, y_min, x_max, y_max)


# def analyze_eye_state(eye_region):
#     if eye_region.size == 0:
#         return 0
    
#     # Calculer le pourcentage de pixels sombres (potentiellement la pupille/iris)
#     threshold = cv2.mean(eye_region)[0] * 0.6  # Seuil adaptatif
#     _, binary_eye = cv2.threshold(eye_region, threshold, 255, cv2.THRESH_BINARY)
    
#     # Calculer le ratio de pixels sombres
#     dark_pixels = np.sum(binary_eye == 0)
#     total_pixels = eye_region.size
#     dark_ratio = dark_pixels / total_pixels
    
#     return dark_ratio

def analyze_eye_state(current_eye_region, previous_eye_region):
    if current_eye_region.size == 0 or previous_eye_region.size == 0:
        return 0, 0
    
    # 1. Détection de changement entre frames
    frame_diff = cv2.absdiff(current_eye_region, previous_eye_region)
    mean_diff = np.mean(frame_diff)
    
    # Normaliser le changement
    current_mean = np.mean(current_eye_region)
    normalized_diff = mean_diff / current_mean if current_mean > 0 else 0
    
    # 2. Analyse de la forme de l'œil
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(current_eye_region, (7, 7), 0)
    
    # Détecter les contours
    _, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculer l'aire totale des contours
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    
    # Normaliser l'aire par rapport à la taille de la région
    normalized_area = total_area / (current_eye_region.shape[0] * current_eye_region.shape[1])
    
    return normalized_diff, normalized_area