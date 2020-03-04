import numpy as np
from skimage import feature
import cv2

dx = [-1, -1, -1, 0, 1, 1, 1, 0]
dy = [-1, 0, 1, 1, 1, 0, -1, -1]

def relocate(x, y, x_boundary, y_boundary):
    if x == 0: x += 1;
    if x == x_boundary - 1: x -= 1
    if y == 0: y += 1
    if y == y_boundary - 1: y -= 1
    return x, y

def LBP(img, landmarks):
    landmarks = landmarks.reshape((-1, 2))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 3
    neighbors = 24
    lbp = feature.local_binary_pattern(img, neighbors, radius, method='uniform')
    extracted_features = []
    for i in range(landmarks.shape[0]):
        # print(landmarks[i][0])
        x = int(landmarks[i][0])
        y = int(landmarks[i][1])
        extracted_features.append(lbp[x][y])
    return np.asarray(extracted_features)




