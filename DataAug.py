import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

# Adding white noise
def add_white_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

# Time stretching
def time_stretch(data, rate=1.5):
    return librosa.effects.time_stretch(data, rate)

# Pitch scaling
def pitch_shift(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# Polarity inversion
def polarity_inversion(data):
    return -data

# Random gain
def random_gain(data, gain_factor=1.5):
    return data * gain_factor       

def plot_waveform(data, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    signal, sr = librosa.load('/Users/nellygarcia/Downloads/WeConnect_Nkanyezi-Mkhize-R.wav')
    noiseSignal = add_white_noise(signal)
    
    # Convert numpy array to 2D tensor (1, N) and ensure it's a float32 tensor
    noiseSignal_tensor = torch.tensor(noiseSignal, dtype=torch.float32).unsqueeze(0)
    
    # Save noiseSignal.wav using the correct tensor
    torchaudio.save("noiseSignal.wav", noiseSignal_tensor, sr)
    
    plot_waveform(noiseSignal, sr, 'Noise Signal')
    # Save plot noiseSignal.png
    plt.savefig("noiseSignal.png")
    
    plot_waveform(signal, sr, 'Original Signal')
    # Save plot OriginalSignal.png
    plt.savefig("OriginalSignal.png")
