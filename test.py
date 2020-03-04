import cv2

img = cv2.imread('data/001/firstFrame.png')
print(img.shape)

import featureExtractor

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (128, 128))
featureExtractor.LBP(img_gray, 1)
