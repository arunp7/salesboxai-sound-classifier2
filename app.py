import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd 
import librosa
import soundfile as sf
import numpy as np
from os import path
import pathlib
from flask import json
from werkzeug.exceptions import HTTPException

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static'
# Check if the upload folder exists and if not create one in the root directory
if(path.exists(UPLOAD_FOLDER) == False):
    os.mkdir(UPLOAD_FOLDER)
    print("Uploads directory created")

ALLOWED_EXTENSIONS = {'wav',"ogg","raw"}

print("Extracting features..")
features_df1 = pd.read_csv("features.csv") 
print("Extracting features done..")

# Create Flask App
app = Flask(__name__)


# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

def convert_to_std_format(file_name):
    f_name = pathlib.Path(file_name).stem
    f_extn = pathlib.Path(file_name).suffix
    
    if( f_extn in [".wav",".mp3"]):
        o_data, o_sr = sf.read(file_name)
        c_file_name = f_name + ".ogg"
        sf.write(c_file_name, o_data, o_sr)        
        return c_file_name
    
    elif (f_extn == ".raw"):
        o_data, o_sr = sf.read(file_name,samplerate=22050,channels = 1, format='RAW',subtype='PCM_32')
        c_file_name = f_name + ".ogg"
        sf.write(c_file_name, o_data, o_sr)        
        return c_file_name
    
    else:
        return file_name


def check_duration(file_name):
    y,sr = librosa.load(file_name)
    dur = librosa.get_duration(y=y, sr=sr)
    if(dur>20):
        return False
    else:
        return True


def get_features(file_name):
    s_file = convert_to_std_format(file_name)
    if s_file:
        X, sample_rate = librosa.load(s_file, mono=True,dtype='float32')

    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)

    if(s_file != file_name):
        os.remove(s_file)        
    return mfccs_scaled
  

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results',
                filename=filename))
    return render_template("index.html")

def file_size_limit(filename):
    size = os.path.getsize(filename)
    if((size*.001) < 1000):
        return True
    else:
        return False

def get_numpy_array(features_df):

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    # encode classification labels
    le = LabelEncoder()
    # one hot encoded labels
    yy = to_categorical(le.fit_transform(y))
    return X,yy,le

def class_label(argument):
    classes = {
        0: "Pressure-Cooker",
        1: "Baby-Cry",
        2: "Ambient-Sound",
        3: "Rain"
    }
    return classes.get(argument, "Unidentified Sound")


def class_label_image(argument):
    classes = {
        0: "PRESSURE-COOKER.jpg",
        1: "BABY-CRY.jpg",
        2: "AMBIENT-SOUND.jpg",
        3: "RAIN.jpg"
    }
    return classes.get(argument, "UNKNOWN.jpg")

# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    size_ok = file_size_limit(filename)
    if(size_ok == True):   
        # Compute audio signal features
        X, y, le = get_numpy_array(features_df1)
        model = load_model("trained_cnn.h5")
        prediction_feature = get_features(filename)
        if(prediction_feature.size != 0):
            prediction_feature = np.expand_dims(np.array([prediction_feature]),axis=2)
            predicted_vector = model.predict_classes(prediction_feature)
            predicted_class = le.inverse_transform(predicted_vector)
            final_pred = class_label(predicted_class[0])
            img_name = class_label_image((predicted_class[0]))
            img_path = os.path.join(app.config['IMAGE_FOLDER'],img_name)
            # Delete uploaded file
            os.remove(filename)
            # Render results
            return render_template("results.html",
                filename=filename,
                detected_sound=img_name
                )
    else:
        return "File size Exceeded"

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))