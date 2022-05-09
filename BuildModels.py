from Plots import SoundFeaturesPlot
from NeuralNet import SirenDetectionNeural
from SVM import SirenDetectionSVM
from RandomForest import SirenDetectionRF
from FeatureExtaction import ExtractingSoundFeaturse
from FilePrepration import FilePreparation
from NeuralNet import SirenDetectionNeural


neural_model_path = "Models/Neural_Model.h5"
get_all_Plots = SoundFeaturesPlot()
extract = ExtractingSoundFeaturse()
files = FilePreparation()
neural = SirenDetectionNeural()
SVM = SirenDetectionSVM()
RandomForest = SirenDetectionRF()
neural_class = SirenDetectionNeural()
# Neural Net ************************************************************************************

# Extreact Data
# extract.save_extracted_features("Dataset", "Neural", 8000)

# Load Files
# files.load_data("ExtractedData/Neural_Model_Data.npy")

# Train Neural
# neural.train_neural_siren("ExtractedData/Neural_Model_Data.npy")

# cross_validation
# neural.cross_validation_neural(
#     "ExtractedData/Neural_Model_Data.npy")

# Grid Search
# neural.grid_search_neural(
#     "ExtractedData/Neural_Model_Data.npy")

# ROC & AUC
# neural.plot_ROC("ExtractedData/Neural_Model_Data.npy")

# Confusion Matrix
# neural.plot_confusion_matrix("ExtractedData/Neural_Model_Data.npy")

# Neural Predict
# predict = neural.predict_neural(
#     "PredictFiles/2.wav", "Models/Neural_Model.h5", sample_rate=8000)
# print("\n\n\n")
# print(f"predict average = {predict}%")

# SVM ******************************************************************************************************************************
# Extreact Data
# extract.save_extracted_features("Dataset", "SVM", 8000)

# Load Files
# files.load_data("ExtractedData/SVM_Model_Data.npy")


# Train SVM
# SVM.train_svm_siren("ExtractedData/SVM_Model_Data.npy")


# Cross Validation
# SVM.cross_validation_SVM(
#     "ExtractedData/SVM_Model_Data.npy")

# Grid Search
# SVM.grid_search_SVM(
#     "ExtractedData/SVM_Model_Data.npy")

# ROC & AUC
# SVM.plot_ROC("ExtractedData/SVM_Model_Data.npy")

# Confusion Matrix
# SVM.plot_confusion_matrix("ExtractedData/SVM_Model_Data.npy")


# #  predict
# predict = SVM.predict_svm(
#     "PredictFiles/4.wav", "Models/SVM_Model.pkl", sample_rate=8000)
# print(f"class = {predict}")


# Random Forest ******************************************************************************************************************************
# Extreact Data
# extract.save_extracted_features("Dataset", "Neural", 8000)

# Load Files
# files.load_data("ExtractedData/Neural_Model_Data.npy")


# Train Random Forest
# RandomForest.train_RF_siren("ExtractedData/Neural_Model_Data.npy")


# cross_validation

# RandomForest.cross_validation_RF(
#     "ExtractedData/Neural_Model_Data.npy")

# Grid Search
# RandomForest.grid_search_RF(
#     "ExtractedData/Neural_Model_Data.npy")

# ROC & AUC
# RandomForest.plot_ROC("ExtractedData/Neural_Model_Data.npy")

# Confusion Matrix
# RandomForest.plot_confusion_matrix("ExtractedData/Neural_Model_Data.npy")

#  predict
predict = RandomForest.predict_RF(
    "PredictFiles/4.wav", "Models/RF_Model.pkl", sample_rate=8000)
print(f"class = {predict}")
