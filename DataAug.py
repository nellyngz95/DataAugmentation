import librosa
print(librosa.__version__)
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
def time_stretch(data, rate=0.8):
   return librosa.effects.time_stretch(data, rate=rate)
    
def plot_waveform(data, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    signal, sr = librosa.load('/Users/nellygarcia/Downloads/WeConnect_Nkanyezi-Mkhize-R.wav')
    
    # Apply time-stretching
    stretchSignal = time_stretch(signal)
    
    # Convert numpy array to 2D tensor (1, N) and ensure it's a float32 tensor
    stretchSignal_tensor = torch.tensor(stretchSignal, dtype=torch.float32).unsqueeze(0)
    
    # Save stretched signal
    torchaudio.save("stretchSignal.wav", stretchSignal_tensor, sr)
    
    # Plot and save waveforms
    plot_waveform(stretchSignal, sr, 'Stretched Signal')
    plt.savefig("stretchSignal.png")
    
    plot_waveform(signal, sr, 'Original Signal')
    plt.savefig("OriginalSignal.png")
