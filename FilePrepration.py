import numpy as np
import math
import librosa
from FeatureExtaction import ExtractingSoundFeaturse


class FilePreparation:

    def load_data(self, extracted_data_path):
        with open(extracted_data_path, 'rb') as f:
            X = np.load(f, allow_pickle=True)
            y = np.load(f, allow_pickle=True)

        print("Data succesfully loaded!")
        print(f'shape of x ={X.shape}')
        print(f'shape of y ={y.shape}')
        return X, y

    def sufficient_segment_number(self, signal_size, sample_rate, default_number):
        number_segment = (int)(signal_size / sample_rate)
        if number_segment > default_number:
            return default_number
        else:
            return number_segment

    def get_features_predict(self, audio, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
        X_predict = []
        filter_data = ExtractingSoundFeaturse()
        num_mfcc_vectors_per_segment = math.ceil(
            sample_rate / hop_length)
        signal, _ = librosa.load(audio, sr=22050)
        filtered_signal = filter_data.butter_bandpass_filter(
            signal)
        size = filtered_signal.size
        for d in range(self.sufficient_segment_number(size, sample_rate, num_segments)):
            start = sample_rate * d
            finish = start + sample_rate
            mfcc = librosa.feature.mfcc(
                filtered_signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                X_predict.append(mfcc.tolist())
            
        return np.array(X_predict)
