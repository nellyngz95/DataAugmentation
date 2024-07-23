import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

# Pitch scaling : pitch factor is number of semi tones to shift. Positive is  higher scale, negative is lower scale
def pitch_shift(data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)
    
def plot_waveform(data, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    print("Pitch Scaling")
    signal, sr = librosa.load('/Users/nellygarcia/Downloads/WeConnect_Nkanyezi-Mkhize-R.wav')
    
    # Apply pitch scaling
    PitchedSignal = pitch_shift(signal,sr)
    
    # Convert numpy array to 2D tensor (1, N) and ensure it's a float32 tensor
    pitchedSignal_tensor = torch.tensor(PitchedSignal, dtype=torch.float32).unsqueeze(0)
    
    # Save stretched signal
    torchaudio.save("PitchedSignal.wav", pitchedSignal_tensor, sr)
    
    # Plot and save waveforms
    plot_waveform(PitchedSignal, sr, 'Pitched Signal')
    plt.savefig("PitchedSignal.png")
    
    plot_waveform(signal, sr, 'Original Signal')
    plt.savefig("OriginalSignal.png")
