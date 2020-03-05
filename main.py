import numpy as np
import dataGenerator
import featureExtractor
import mySDM
import cv2
import matplotlib.pyplot as plt

# todo: data preprocessing
images, annots = dataGenerator.readAndScaleVideo1()
img = images[1]

# todo: training phase
sdm = mySDM.mySDM(n_regressors=3)
sdm.fit(images, annots)

landmarks = sdm.predict(img)

# todo: visualize
plt.imshow(img)
for i in range(landmarks.shape[0]):
    plt.scatter(landmarks[i][0], landmarks[i][1], color='g')
plt.show()








