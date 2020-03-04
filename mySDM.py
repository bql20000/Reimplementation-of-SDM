import numpy as np
from sklearn.neural_network import MLPRegressor
import featureExtractor
import cv2

class mySDM:
    def __init__(self, n_regressors=10, N=None, regressors=None, initialization=None, feature_extractor='LBP', n_features=None, n_landmarks=68):
        self.n_regressors = n_regressors
        self.N = N
        self.regressors = regressors
        self.initialization = initialization
        self.feature_extractor = feature_extractor
        if feature_extractor == 'LBP': self.n_features = 68
        self.n_landmarks = n_landmarks

    def extract_feature(self, img, x):
        if self.feature_extractor == 'LBP':
            return featureExtractor.LBP(img, x)

    def fit(self, images, annots):
        self.N = images.shape[0]
        annots = annots.reshape((self.N, -1))
        # print(annots.shape)
        self.regressors = []
        self.initialization = annots[0]

        for i in range(self.n_regressors):
            self.regressors.append(MLPRegressor((2 * self.n_landmarks,), activation='identity', solver='sgd', batch_size=self.N, max_iter=1000, verbose=True, learning_rate_init=0.0001, tol=0.001, momentum=0, n_iter_no_change=3))

        X = np.ones((self.N, 2 * self.n_landmarks))
        for i in range(self.N): X[i] = self.initialization

        for k in range(self.n_regressors):
            # extract feature at key points
            featured_vectors = []
            for i in range(self.N):
                featured_vectors.append(self.extract_feature(images[i], X[i]))
            featured_vectors = np.asarray(featured_vectors).reshape((self.N, self.n_features))

            # train k_th regressor
            # print(featured_vectors.shape, annots.shape)
            featured_vectors /= np.amax(featured_vectors)
            self.regressors[k].fit(featured_vectors, annots - X)
            delta_x = self.regressors[k].predict(featured_vectors)
            X += delta_x

    def predict(self, img):
        img = cv2.resize(img, (256, 256))
        x = self.initialization
        for k in range(self.n_regressors):
            fv = self.extract_feature(img, x)
            delta_x = self.regressors[k].predict(fv.reshape(1, -1))
            x += delta_x[0]
        return x.reshape((-1, 2))

    def load_model(self, path):
        return 0

    def save_model(self, folder=''):
        return 0