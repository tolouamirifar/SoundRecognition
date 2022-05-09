import matplotlib.pyplot as plt
from FeatureExtaction import ExtractingSoundFeaturse
import librosa.display
import librosa


class SoundFeaturesPlot:

    # Plot Waves #################################################################################################################
    def plot_wave(self, audio_path, audio_name):
        features = ExtractingSoundFeaturse()
        # fig = plt.figure(figsize=(12, 5))
        signal, sr = librosa.load(audio_path)
        librosa.display.waveplot(signal, alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.title(f"Wave for {audio_name}")
        # plt.savefig(f'samples_plots/{audio_name}.png')
        # plt.close(fig)

    def plot_mfccs(self, audio_path, audio_name):
        features = ExtractingSoundFeaturse()
        # fig = plt.figure(figsize=(13, 5))
        signal, sr = librosa.load(audio_path)
        mfcc = features.get_mfccs(signal, 13, 2048, 512)
        librosa.display.specshow(
            mfcc, sr=sr, x_axis="time")
        plt.colorbar(format="%+2.f")
        plt.xlabel("Time")
        plt.ylabel("MFCCs Frequency")
        plt.title(f"MFCCs for {audio_name}")

    def plot_all_features(self, audio_path, audio_name):
        fig = plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        self.plot_wave(audio_path, audio_name)
        plt.subplot(2, 1, 2)
        self.plot_mfccs(audio_path, audio_name)
        plt.tight_layout()
        plt.savefig(f'Plots/PlotResult.png')

    # def plot_all_waves(self, audiofiles):
    #     fig = plt.figure(figsize=(12, 5))
    #     for index, audio in enumerate(audiofiles):
    #         fig.add_subplot(len(audiofiles), 1, index + 1)
    #         librosa.display.waveplot(audio, alpha=0.5)
    #     plt.xlabel("Time")
    #     plt.ylabel("Power")
    #     plt.title("Wave of The Siren")
    #     # plt.savefig('samples_plots/wave_plot.png')

    # # Plot Waves ###################################################################################################################
    # def plot_amplitude_envelope(self, signal_specified, ae_specified, t):
    #     fig = plt.figure(figsize=(12, 5))
    #     fig.add_subplot(1, 1, 1)
    #     librosa.display.waveplot(signal_specified, alpha=0.5)
    #     plt.plot(t, ae_specified, color='r')
    #     plt.xlabel("Time")
    #     plt.ylabel("Amplitude")
    #     plt.title("Amplitude Envelope of The Siren")
    #     # plt.savefig('samples_plots/wave_plot.png')

    # def plot_rms(self, signal_specified, rms_specified, t):
    #     # fig = plt.figure()
    #     # fig.add_subplot(1, 1, 1)
    #     librosa.display.waveplot(signal_specified, alpha=0.5)
    #     plt.plot(t, rms_specified, color='r')
    #     plt.xlabel("Time")
    #     plt.ylabel("RMS")
    #     plt.title("Root Mean Squar of The Siren")
    #     # plt.savefig('samples_plots/wave_plot.png')

    # def plot_zcr(self, signal_specified, zcr_specified, t):
    #     # fig = plt.figure()
    #     # fig.add_subplot(1, 1, 1)
    #     librosa.display.waveplot(signal_specified, alpha=0.5)
    #     plt.plot(t, zcr_specified, color='r')
    #     # plt.savefig('samples_plots/wave_plot1.png')
    #     plt.show()

    # def plot_fft(self, fft_frequency, fft_magnitude):
    #     # fig = plt.figure(figsize=(12, 5))
    #     plt.plot(fft_frequency, fft_magnitude)
    #     plt.xlabel("Frequency of Signal")
    #     plt.ylabel("Magnitude of the Frequency")
    #     plt.title("fft of the Siren")

    # def plot_stft(self, stft_log_abs, sr_specified, Hop_Length, Y_Axis="linear"):
    #     # fig = plt.figure(figsize=(13, 6))
    #     librosa.display.specshow(
    #         stft_log_abs, sr=sr_specified, hop_length=Hop_Length, x_axis="time", y_axis=Y_Axis)
    #     plt.colorbar(format="%+2.f")
    #     plt.xlabel("Time")
    #     plt.ylabel("STFC Frequency")
    #     plt.title("STFT of the siren")
    #     # plt.show()

    # def plot_mel_frequency(self, mel_log_abs, sr_specified, Hop_Length, Y_Axis="linear"):
    #     # fig = plt.figure(figsize=(13, 6))
    #     librosa.display.specshow(
    #         mel_log_abs, sr=sr_specified, hop_length=Hop_Length, x_axis="time", y_axis=Y_Axis)
    #     plt.colorbar(format="%+2.f")
    #     plt.xlabel("Time")
    #     plt.ylabel("Mel Frequency")
    #     plt.title("Mel Frequency of the siren")
