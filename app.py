from flask import Flask, request, redirect, render_template
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1280 * 1280


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploaded', f.filename)
        f.save(file_path)
        return 'abc'

if __name__ == '__main__':
    app.run(debug=True)


