from re import VERBOSE
import matplotlib.pyplot as plt
# For cross_val_score, GridSearchCV, RandomizedSearchCV
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
import pickle
import math
import numpy as np
import os
import json
import itertools
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from FeatureExtaction import ExtractingSoundFeaturse
from FilePrepration import FilePreparation
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class SirenDetectionNeural:
    def build_classifier(self):
        # build network topology
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data("ExtractedData/Neural_Model_Data.npy")
        model = keras.Sequential()
        model.add(keras.layers.Flatten(
            input_shape=(X.shape[1], X.shape[2])))
        model.add(keras.layers.Dense(512, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(256, activation='relu',))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation='relu',))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

        return model

    def evaluation_summery(self, model, X_train, y_train, X_test, y_test):

        loss_train, acc_train, precision_train, recall_train = model.evaluate(
            X_train, y_train, verbose=0)
        loss_test, acc_test, precision_test, recall_test = model.evaluate(
            X_test, y_test, verbose=0)
        print('Accuracy:')
        print(f"\ttrain:{(acc_train * 100 ):0.2f}%")
        print(f"\ttest:{(acc_test * 100 ):0.2f}%")

        print('Precision:')
        print(f"\train:{(precision_train * 100 ):0.2f}%")
        print(f"\ttest:{(precision_test * 100 ):0.2f}%")

        print('Recall:')
        print(f"\ttrain:{(recall_train * 100 ):0.2f}%")
        print(f"\ttest:{(recall_test * 100 ):0.2f}%")

    def train_neural_siren(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        model = self.build_classifier()
        class_weights = {0: 1, 1: 2}
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=300, batch_size=64, verbose=0, class_weight=class_weights)
        model.save("Models/Neural_Model.h5")
        self.evaluation_summery(model, X_train, y_train, X_test, y_test)
        self.plot_history(history)

    def predict_neural(self, sound_path, model_name, sample_rate):
        files = FilePreparation()
        predict = []
        X_predict = files.get_features_predict(
            sound_path, sample_rate, num_segments=10)

        loaded_model = keras.models.load_model(model_name)
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        loaded_model.compile(optimizer=optimiser,
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
        all_segment_predict = loaded_model.predict(X_predict)
        for segment in all_segment_predict:
            predict.append(round(float(segment), 2))
        average_predict = round(float(np.average(np.array(predict))), 2)
        print(np.array(X_predict).shape)
        return average_predict

    def plot_history(self, history):
        plt.plot(history.history["accuracy"], label="Train Acuracy")
        plt.plot(history.history["val_accuracy"], label="Test Acuracy")
        plt.savefig(f'Plots/Accuracy_epoches.png')
        plt.show()

    def cross_validation_neural(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        class_weights = {0: 1, 1: 2}
        model = KerasClassifier(
            build_fn=self.build_classifier, epochs=300, batch_size=64, verbose=0)
        model.fit(X_train, y_train)
        # kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
        # for i, (train, test) in enumerate(kfold.split(X, y)):
        #     model.fit(X[train], y[train],
        #               epochs=300, batch_size=64, verbose=0, class_weight=class_weights)
        #     scores = model.evaluate(X[test], y[test], verbose=0)
        #     print(f"Held out test accuracy fold-{i+1}:{100 * scores[1]:0.2f}%")
        for fold in range(2, 9):
            score_train = sklearn.model_selection.cross_val_score(
                model, X=X_train, y=y_train, cv=fold, n_jobs=-1).mean()
            score_test = sklearn.model_selection.cross_val_score(
                model, X=X_test, y=y_test, cv=fold, n_jobs=-1).mean()
            print(
                f"held-out train accuracy ({fold}-fold):\t {100 * score_train:0.2f}%")
            print(
                f"held-out test accuracy ({fold}-fold):\t {100 * score_test:0.2f}%")
            print(f"fold={fold}")

    def grid_search_neural(self, extracted_data_path):

        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        model = KerasClassifier(
            build_fn=self.build_classifier, verbose=0)
        batch_size = [15, 32, 64]
        epochs = [150, 200, 250]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = sklearn.model_selection.GridSearchCV(
            estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_train, y_train)
        print("\n\nBest: %f using %s" %
              (grid_result.best_score_, grid_result.best_params_))

    def plot_ROC(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        class_weights = {0: 1, 1: 2}
        model = self.build_classifier()
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=300, batch_size=64, verbose=0, class_weight=class_weights)
        y_pred_keras = model.predict(X_test).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(
            y_test, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        print('AUC=%.2f%%' % (auc_keras*100))
        plt.plot(fpr_keras, tpr_keras, marker='.',
                 label=f"Neural Classifier (AUC ={100 * auc_keras:0.1f} )")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Neural Net Model')
        plt.legend()
        plt.savefig(f'Plots/ROC_Neural.png')
        plt.show()

    def plot_confusion_matrix(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        class_weights = {0: 1, 1: 2}
        model = self.build_classifier()
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=300, batch_size=64, verbose=0, class_weight=class_weights)
        y_pred_keras = model.predict(X_test).ravel()
        y_pred_scaled = np.around(y_pred_keras)
        classes = ["0", "1"]
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred_scaled)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confiusion Matrix for Neural Nodel")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.savefig(f'Plots/Confusion_Matrix_Neural.png')
        plt.show()
