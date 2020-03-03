import cv2
import os

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


extractFirstFrameEachVideo()