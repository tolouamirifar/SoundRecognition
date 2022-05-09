import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.svm               # For SVC
import sklearn.tree              # For DecisionTreeClassifier
import sklearn.metrics           # For accuracy_score
from sklearn.model_selection import train_test_split
from FilePrepration import FilePreparation
from sklearn.metrics import precision_score, recall_score, accuracy_score, plot_roc_curve, plot_confusion_matrix


class SirenDetectionRF:
    def build_forest(self):
        model = sklearn.ensemble.RandomForestClassifier(
            random_state=50, n_estimators=1000, min_samples_leaf=5, max_depth=20,
            max_features='sqrt', class_weight={1: 2, 0: 1}, n_jobs=-1)
        return model

    def evaluation_summery(self, model, X_train, y_train, X_test, y_test):

        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        precision_train = precision_score(
            y_train, y_train_predict, average='binary')
        precision_test = precision_score(
            y_test, y_test_predict, average='binary')
        print('Precision:')
        print(f"train:{(precision_train * 100 ):0.1f}%")
        print(f"test:{(precision_test * 100 ):0.1f}%")

        recall_train = recall_score(y_train, y_train_predict, average='binary')
        recall_test = recall_score(y_test, y_test_predict, average='binary')
        print('Recall:')
        print(f"train:{(recall_train * 100 ):0.1f}%")
        print(f"test:{(recall_test * 100 ):0.1f}%")

        accuracy_train = accuracy_score(y_train, y_train_predict)
        accuracy_test = accuracy_score(y_test, y_test_predict)
        print('Accuracy:')
        print(f"train:{(accuracy_train * 100 ):0.1f}%")
        print(f"test:{(accuracy_test * 100 ):0.1f}%")

    def train_RF_siren(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_reshaped = X.reshape(len(X), X.shape[1] * X.shape[2])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2)
        model = self.build_forest()
        model.fit(X_train, y_train)
        self.evaluation_summery(model, X_train, y_train, X_test, y_test)
        pickle.dump(model, open("Models/RF_Model.pkl", 'wb'))

    def predict_RF(self, sound_path, model_name, sample_rate):
        files = FilePreparation()
        predict = []
        X_predict = files.get_features_predict(
            sound_path, sample_rate, num_segments=10)
        loaded_model = pickle.load(
            open(model_name, 'rb'))
        for segment in X_predict:
            segment_reshaped = segment.flatten().reshape(1, -1)
            predict.append(loaded_model.predict(
                segment_reshaped))
        average_predict = round(float(np.average(np.array(predict))), 2)
        return average_predict

    def cross_validation_RF(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_reshaped = X.reshape(len(X), X.shape[1] * X.shape[2])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2)
        model = self.build_forest()
        model.fit(X_train, y_train)
        for fold in range(2, 9):
            score_train = sklearn.model_selection.cross_val_score(
                model, X=X_train, y=y_train, cv=fold, n_jobs=-1).mean()
            score_test = sklearn.model_selection.cross_val_score(
                model, X=X_test, y=y_test, cv=fold, n_jobs=-1).mean()
            print(
                f"held-out train accuracy ({fold}-fold):\t {100 * score_train:0.2f}%")
            print(
                f"held-out test accuracy ({fold}-fold):\t {100 * score_test:0.2f}%")

    def grid_search_RF(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_reshaped = X.reshape(len(X), X.shape[1] * X.shape[2])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2)
        model = self.build_forest()
        model.fit(X_train, y_train)
        random_state = [5, 10, 20, 30]
        min_samples_leaf = [5, 10, 20, 30]
        max_depths = [1, 5, 10, 15, 20]
        param_grid = dict(random_state=random_state,
                          max_depth=max_depths, min_samples_leaf=min_samples_leaf)
        grid = sklearn.model_selection.GridSearchCV(
            estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_test, y_test)
        print("\n\nBest: %f using %s" %
              (grid_result.best_score_, grid_result.best_params_))

    def plot_ROC(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_reshaped = X.reshape(len(X), X.shape[1] * X.shape[2])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2)
        model = self.build_forest()
        model.fit(X_train, y_train)
        plot_roc_curve(model, X_test.reshape(-1, 208), y_test)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Random Forest Model')
        plt.savefig(f'Plots/ROC_RandomForest.png')
        plt.show()

    def plot_confusion_matrix(self, extracted_data_path):
        extracted_files = FilePreparation()
        X, y = extracted_files.load_data(extracted_data_path)
        X_reshaped = X.reshape(len(X), X.shape[1] * X.shape[2])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2)
        model = self.build_forest()
        model.fit(X_train, y_train)
        plot_confusion_matrix(model, X_test.reshape(-1, 208), y_test,
                              cmap=plt.cm.Blues,)
        plt.title('Confusion Matrix for Forest Model')
        plt.savefig(f'Plots/Confusion_Matrix_Forest.png')
        plt.show()
