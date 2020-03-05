import cv2
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np

def readAllFrames(url):
    vid = cv2.VideoCapture(url)
    count = 0
    ret = 1
    while ret:
        ret, img = vid.read()
        count += 1
        print(count)
        cv2.imwrite('data/001/frames_from_opencv/frame%d.png' % count, img)
    return count


def extractFirstFrameEachVideo():
    path = 'data'
    count = 0
    for foldername in os.listdir(path):
        vid = cv2.VideoCapture(path + '/' + foldername + '/' + 'vid.avi')
        ret, firstFrame = vid.read()
        cv2.imwrite(path + '/' + foldername + '/' + 'firstFrame.png', firstFrame)


def readAndScaleData():
    images = []
    annots = []
    path = 'data'
    count = 0
    img_size = 256      # 256 x 256
    for foldername in os.listdir(path):
        img = cv2.imread(path + '/' + foldername + '/' + 'firstFrame.png')
        annot = pandas.read_csv(path + '/' + foldername + '/' + 'annot/' + '000001.pts')['version: 1'][2:-1]
        height = img.shape[0]
        width = img.shape[1]
        images.append(cv2.resize(img, (img_size, img_size)))
        list_annot = []
        X = []
        Y = []

        for ss in annot:
            x, y = ss.split(' ')
            x = float(x) * img_size / width
            y = float(y) * img_size / height
            list_annot.append([x, y])
            X.append(x)
            Y.append(y)
        annots.append(np.asarray(list_annot))
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        if (count == 11):
            plt.imshow(img)
            plt.scatter(X, Y)
            plt.show()
        """
    return np.asarray(images), np.asarray(annots)


def readAndScaleVideo1():
    images = []
    annots = []
    frame_folder = 'data/001/frames_from_opencv/'
    annot_foler = 'data/001/annot/'
    img_size = 256      # 256 x 256
    height = 720
    width = 1280
    count = 0
    n_images = 300
    for filename in os.listdir(frame_folder):
        img = cv2.imread(frame_folder + '/' + filename)
        images.append(cv2.resize(img, (img_size, img_size)))
        count += 1
        print(count)
        if (count == 1): plt.imshow(images[-1])
        if (count > n_images): break
    count = 0
    for filename in os.listdir(annot_foler):
        annot = pandas.read_csv(annot_foler + '/' + filename)['version: 1'][2:-1]
        list_annot = []
        count += 1
        print(count)
        for ss in annot:
            x, y = ss.split(' ')
            x = float(x) * img_size / width
            y = float(y) * img_size / height
            list_annot.append([x, y])
        annots.append(np.asarray(list_annot))
        if (count > n_images): break
    return np.asarray(images), np.asarray(annots)

# -------------------------------------------- MAIN --------------------------------------------------------------------
# readAllFrames('data/001/vid.avi')     # done
# extractFirstFrameEachVideo()          # done

# images, annots = readAndScaleData()
