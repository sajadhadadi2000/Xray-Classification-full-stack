import os
import requests
from flask import Flask, request, render_template, redirect
import pickle
from werkzeug.utils import secure_filename 

app = Flask(__name__)

# Define the path to the directory
upload_folder = os.path.join(os.getcwd(), 'static')

# Create the directory if it doesn't exist
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Configure the app to use this directory for image uploads
app.config['IMAGE_UPLOADS'] = upload_folder


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    if request.method == 'POST':
        print(os.getcwd()) 
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        res = requests.post("http://torchserve-mar:8080/predictions/xray", files={'data': open(img_path, 'rb')})
        prediction = res.json()

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
