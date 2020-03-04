import numpy as pd
import dataGenerator
import featureExtractor
import mySDM
import cv2
import visualization

# todo: data preprocessing
images, annots = dataGenerator.readAndScaleVideo1()

print(annots.shape)

# todo: training phase
sdm = mySDM.mySDM(n_regressors=1)
sdm.fit(images, annots)
sdm.save_model(folder='trained_SDM')

img = images[2]
ann = sdm.predict(img)

visualization.visualize(img, ann)



