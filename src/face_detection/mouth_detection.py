from face_detection.utils import euclaideanDistance



def mouthRatio(landmarks, lips_indices):
    # Points du haut et du bas de la bouche (verticaux)
    top_point = landmarks[lips_indices['top'][0]]
    bottom_point = landmarks[lips_indices['bottom'][0]]
    
    # Points gauche et droit (horizontaux) pour normalisation
    left_point = landmarks[lips_indices['left'][0]]
    right_point = landmarks[lips_indices['right'][0]]
    
    # Distance verticale entre les lèvres
    vertical_distance = euclaideanDistance(top_point, bottom_point)
    
    # Distance horizontale pour normalisation
    horizontal_distance = euclaideanDistance(left_point, right_point)
    
    # Calcul du ratio normalisé
    ratio = vertical_distance / horizontal_distance
    
    return ratio