from Plots import SoundFeaturesPlot
from NeuralNet import SirenDetectionNeural
from SVM import SirenDetectionSVM
from RandomForest import SirenDetectionRF
from FeatureExtaction import ExtractingSoundFeaturse
from FilePrepration import FilePreparation
from flask_cors import CORS, cross_origin


import os
from flask import Flask, make_response, send_from_directory, send_file, flash, request, redirect, url_for, session
from flask_cors import CORS, cross_origin
# matplotlib.use('TKAgg')


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# save dataa nd model for siren detection ******************************************************************************************************************************
get_all_Plots = SoundFeaturesPlot()
extract = ExtractingSoundFeaturse()
files = FilePreparation()
neural = SirenDetectionNeural()
SVM = SirenDetectionSVM()
RandomForest = SirenDetectionRF()
neural_class = SirenDetectionNeural()

clientURL = 'https://emergencysirendetection.ca'


@app.route('/get_plots', methods=['POST'])
@cross_origin(origin=clientURL, headers=['Content-Type', 'Authorization'])
def get_plots():
    target = os.path.join('PredictFiles')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    print(filename)
    get_all_Plots.plot_all_features(destination, filename)
    returned_wavePlot = send_from_directory(
        directory="Plots", filename="PlotResult.png")
    return returned_wavePlot


@app.route('/predict_neural', methods=['POST'])
@cross_origin(origin=clientURL, headers=['Content-Type', 'Authorization'])
def predict_neural():
    target = os.path.join('PredictFiles')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    predict = neural_class.predict_neural(
        destination, "Models/Neural_Model.h5", sample_rate=8000)
    os.remove(destination)
    return f"{predict}"


@app.route('/predict_forest', methods=['POST'])
@cross_origin(origin=clientURL, headers=['Content-Type', 'Authorization'])
def predict_RF():
    target = os.path.join('PredictFiles')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    predict = RandomForest.predict_RF(
        destination, "Models/RF_Model.pkl", sample_rate=8000)
    os.remove(destination)
    return f"{predict}"


@app.route('/predict_svm', methods=['POST'])
@cross_origin(origin=clientURL, headers=['Content-Type', 'Authorization'])
def predict_svm():
    target = os.path.join('PredictFiles')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    predict = SVM.predict_svm(
        destination, "Models/SVM_Model.pkl", sample_rate=8000)
    os.remove(destination)
    return f"{predict}"


# Run Server ****************************************************************************************
if __name__ == '__main__':
    app.run()
