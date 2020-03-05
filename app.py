from flask import Flask, request, redirect, render_template
import matplotlib.pyplot as plt
import dataGenerator
import mySDM

# todo: generate data
images, annots = dataGenerator.readAndScaleVideo1()

# todo: training phase
sdm = mySDM.mySDM(n_regressors=1)
sdm.fit(images, annots)

def add_landmarks_and_save(img):
    result_path = 'demo/result.png'
    landmarks = sdm.predict(img)
    plt.show(img)
    for i in range(landmarks.shape[0]):
        plt.scatter(landmarks[i][0], landmarks[i][1], color='g')
    plt.axes.get_xaxis().set_visible(False)
    plt.axes.get_yaxis().set_visible(False)
    img.save(result_path)
    return img


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    for img in request.files.getlist("file"):
        img.save('demo/test.png')
        add_landmarks_and_save(img)
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)





