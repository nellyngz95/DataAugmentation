import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

#Gain factor 
def random_gain(data, gain_factor=1.5):
    return data * gain_factor  
    
def plot_waveform(data, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    print("Random gain")
    signal, sr = librosa.load('/Users/nellygarcia/Downloads/WeConnect_Nkanyezi-Mkhize-R.wav')
    
    # Apply pitch scaling
    GainedSignal = random_gain(signal)
    
    # Convert numpy array to 2D tensor (1, N) and ensure it's a float32 tensor
    GainedSignal_tensor = torch.tensor(GainedSignal, dtype=torch.float32).unsqueeze(0)
    
    # Save stretched signal
    torchaudio.save("GainedSignal.wav", GainedSignal_tensor, sr)
    
    # Plot and save waveforms
    plot_waveform(GainedSignal, sr, 'Gained Signal')
    plt.savefig("GainedSignal.png")
    
    plot_waveform(signal, sr, 'Original Signal')
    plt.savefig("OriginalSignal.png")