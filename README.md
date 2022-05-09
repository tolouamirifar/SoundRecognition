# SoundRecognition
## Subtitle: Traffic Management for Emergency Vehicle Priority Based on Siren Sound Recognition
This project aims to design a system which passes emergency vehicles based on detecting their siren sound pattern in emergency situations. The system's goal is to apply a supervised learning algorithm that identifies siren sound patterns, direction, and distance of an emergency vehicle, to generate a signal for the traffic light, and prioritize relieving traffic flow in the direction that the emergency vehicle exists.
## Data Description:
Modality: Waveform Audio File (WAV)

![image](https://user-images.githubusercontent.com/73673501/167509481-7d887f9f-37ca-443d-8ae9-96574586af2f.png)

Input: The original data are waveforms. Features extracted from waveforms that will be fed to the classifier algorithm can be analyzed in the time domain (e.g. RMSE of waveform), the frequency domain (e.g. Amplitude of individual frequencies), perceptual features (e.g. Mel-Frequency Cepstral Coefficients (MFCCs)), and windowing features (e.g. Hamming distances of windows). 

![image](https://user-images.githubusercontent.com/73673501/167509510-e628f96b-0544-4eae-8f8f-58f92327b9e5.png)

Output: The output will be the result of classifying soundwaves to emergency vehicle siren or non-emergency noise.
