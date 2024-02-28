"""
    Generate and save melspectrograms for an audio file
"""
import librosa
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

# Load audio file using librosa
def load_audio_file_custom(file_path:str, offset:Optional[float]=None, duration:Optional[float]=None):
    try:
        y, sr = librosa.load(
            file_path,
            offset=float(offset),
            duration=float(duration)
        )
        return y, sr
    except Exception as e:
        print(f"Error while loading {file_path}: {e}")
        return None, None

# Get duration of audio file
def duration(file_path:str) -> Optional[float]:
    try:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(
            y=y,
            sr=sr
        )
        return float(duration)
    except Exception as e:
        print(f"Error while computing duration for {file_path}: {e}")
        return None

# Generate and save melspectrogram
def generate_melspectrogram(save_file_path:str, y:np.ndarray, sr:int, n_mels:int=128, hop_length:int=512) -> Optional[str]:
    try:
        # Generate mel spectrogram
        melspectrum = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length= hop_length,
            window='hann',
            n_mels=n_mels
        )
        # Save image
        # @note ref: https://www.tutorialspoint.com/how-to-save-a-librosa-spectrogram-plot-as-a-specific-sized-image#:~:text=Compute%20a%20mel%2Dscaled%20spectrogram,the%20img%20using%20savefig().
        S_dB = librosa.power_to_db(melspectrum, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr)
        plt.savefig(save_file_path, bbox_inches="tight",pad_inches=-0.1) # Removing whitespace ref: https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image 
        return save_file_path
    except Exception as e:
        print(f"Error while saving melspectrogram to {save_file_path}: {e}")
        return None