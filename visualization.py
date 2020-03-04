import matplotlib.pyplot as plt


def visualize(img, landmarks):
    plt.imshow(img)
    for i in range(landmarks.shape[0]):
        plt.scatter(landmarks[i][0], landmarks[i][1])
    plt.show()