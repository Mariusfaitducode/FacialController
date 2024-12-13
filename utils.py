from math import sqrt

# Euclaidean distance to calculate the distance between the two points
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance