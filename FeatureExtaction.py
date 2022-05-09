import librosa
import math
import os
import librosa.display
import numpy as np
from werkzeug.utils import secure_filename
# from classes_plots.sound_features_plots import SoundFeaturesPlot


class ExtractingSoundFeaturse:

    def load_signals(self, audio_files):
        audios = []
        sr = []
        for value in audio_files:
            extract_audios, extract_sr = librosa.load(value)
            audios.append(extract_audios)
            sr.append(extract_sr)
        return audios, sr

    def get_frame_to_time(self, ae_audio, hop_length):
        frames = range(0, ae_audio.size)
        t = librosa.frames_to_time(frames, hop_length=hop_length)
        return t, frames

    # Amplitude **************************************************************************************************************

    def get_amplitude_envelope(self, load_audio, frame_size, hop_length):
        amplitude_envelope = []
        for i in range(0, len(load_audio), hop_length):
            current_frame_amplitude_envelope = max(load_audio[i:i+frame_size])
            amplitude_envelope.append(current_frame_amplitude_envelope)
        return np.array(amplitude_envelope)

    # RMS Energy **************************************************************************************************************

    def rms_energy(self, audio, frame_size, hop_length):
        return librosa.feature.rms(audio, frame_length=frame_size, hop_length=hop_length)[0]

    # zero crossing rate **************************************************************************************************************
    def Zro_crossing_rate(self, signal, frame_size, hop_length):
        return librosa.feature.zero_crossing_rate(signal, frame_length=frame_size, hop_length=hop_length)[0]

    # RMS Energy **************************************************************************************************************
    def get_stft(self, load_files, Frame_Size, Hop_Leangth):
        stft = librosa.core.stft(
            load_files, n_fft=Frame_Size, hop_length=Hop_Leangth)
        return stft

    # fft **************************************************************************************************************
    # def get_fft(self, signals_specific, sr_specific, F_Ratio):
    #     fft_signal = sp.fft.fft(signals_specific)
    #     fft_magnitude = np.abs(fft_signal)
    #     frequency = np.linspace(0, sr_specific, len(fft_magnitude))
    #     num_frequency_bins = int(len(frequency) * F_Ratio)
    #     return frequency[0:num_frequency_bins], fft_magnitude[0:num_frequency_bins]

    # Mel Spectogram  (MFS) **************************************************************************************************************
    def get_log_scale(self, signal_spectrum):
        log_scale = librosa.power_to_db(signal_spectrum)
        return log_scale

    def get_mel_filterbanks(self,  Frame_Size, sr, N_Mels):
        log_scale = librosa.filters.mel(n_fft=Frame_Size, sr=sr, n_mels=N_Mels)
        return log_scale

    def get_mel_frequency(self, signal_spectrum, Frame_Size, sr, N_Mels, Hop_Length):
        log_scale = librosa.feature.melspectrogram(
            signal_spectrum, n_fft=Frame_Size, sr=sr, n_mels=N_Mels, hop_length=Hop_Length)
        return log_scale

    # Mel Frequency Cepstral Coefficient (MFCC) **************************************************************************************************************
    def get_mfccs(self, signal, N_MFCC, N_FFT, HOP_LEBGTH, sr=22050):
        mfccs = librosa.feature.mfcc(
            signal, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEBGTH, sr=sr)
        return mfccs

    # Band Pass **********************************************************************************************************************************************
    def butter_bandpass_filter(self, signal, lowcut=500, highcut=1850, fs=8000, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal)
        return y

    def sufficient_segment_number(self, signal_size, sample_rate, default_number):
        number_segment = (int)(signal_size / sample_rate)
        if number_segment > default_number:
            return default_number
        else:
            return number_segment

    def save_extracted_features(self, dataset_path, for_model, sample_rate=8000, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
        X = []
        y = []

        num_mfcc_vectors_per_segment = math.ceil(
            sample_rate / hop_length)
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            if dirpath is not dataset_path:
                print(dirpath)
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    signal, _ = librosa.load(file_path, sr=22050)
                    filtered_signal = self.butter_bandpass_filter(
                        signal)
                    signal_size = filtered_signal.size
                    for index in range(self.sufficient_segment_number(signal_size, sample_rate, num_segments)):
                        start = sample_rate * index
                        finish = start + sample_rate
                        mfcc = librosa.feature.mfcc(
                            filtered_signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            if(i == 1):
                                X.append(mfcc.tolist())
                                y.append(1)
                            elif (i == 2):
                                X.append(mfcc.tolist())
                                if(for_model == "SVM"):
                                    y.append(-1)
                                else:
                                    y.append(0)

        print("Data successfuly extracted!")

        with open(f"ExtractedData/{for_model}_Model_Data.npy", 'wb',) as f:
            np.save(f, np.array(X), allow_pickle=True)
            np.save(f, np.array(y), allow_pickle=True)

        print("Data successfuly saved!")
