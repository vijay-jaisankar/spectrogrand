"""
    Collection of helper functions pertaining to the spectrogram domain of spectrogrand
"""
from typing import Optional, List
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
torch.random.manual_seed(42)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

from audio_helpers import load_wav_file, load_wav_chunk

IDX_TO_LABEL_MAPPING = {0:'future house', 1:'bass house', 2:'progressive house', 3:'melodic house'}

"""
    @method generate_and_save_melspectrogram
        Generate and save a melspectrogram for audio data 
    @param input_data: Numpy array containing the audio to generate a spectrogram for
    @param input_sampling_rate: Sampling rate for the input audio
    @param output_file_path: Path to the output file where the generated melspectrogram is to be stored
    @param n_mels: Number of buckets used in the melspectrogram computation (default:128)
    @param hop_length: Hop length used in the melspectrogram computation (default: 512)
"""
def generate_and_save_melspectrogram(input_data:np.ndarray, input_sampling_rate:int, output_file_path:str, n_mels:int=128, hop_length:int=512) -> Optional[str]:
    try:
        # Generate melspectrogram
        melspectrum = librosa.feature.melspectrogram(
            y=input_data,
            sr=input_sampling_rate,
            hop_length= hop_length,
            window='hann',
            n_mels=n_mels
        )
        S_dB = librosa.power_to_db(melspectrum, ref=np.max)

        # Save image
        img = librosa.display.specshow(S_dB, sr=input_sampling_rate)
        plt.savefig(output_file_path, bbox_inches="tight",pad_inches=-0.1) # Removing whitespace ref: https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image 
        return output_file_path
    except Exception as e:
        print(f"Error while generating and saving melspectrogram: {e}")
        return None
    
"""
    @method generate_and_save_melspectrogram_stream
        Generate and save contiguous melspectrograms given a parent audio file
    @param input_file_path: Path to the input audio file
    @param output_dir: Path to the parent directory where the contiguous melspectrogram images are to be stored
    @param chunk_duration: Length of the contiguous chunks of the audio (default:0.1)
"""
def generate_and_save_melspectrogram_stream(input_file_path:str, output_dir:str, chunk_duration:float=0.1) -> Optional[List[str]]:
    try:
        # Load the audio file to get its duration
        parent_sr, parent_data = load_wav_file(input_file_path=input_file_path)
        parent_duration = librosa.get_duration(y=parent_data,sr=parent_sr)

        # Keep running count of the current time index and number of images generated
        num_images_generated = 0
        saved_output_file_names = []

        current_chunk_time_start = 0.0
        current_chunk_time_end = current_chunk_time_start + chunk_duration

        while float(current_chunk_time_end) <= float(parent_duration):
            output_file = f"{output_dir}/melspec_{num_images_generated}.png"
            # Load chunk data
            sr, y = load_wav_chunk(input_file_path=input_file_path,chunk_offset=current_chunk_time_start,chunk_duration=chunk_duration)
            # Construct melspectrogram
            output_file = generate_and_save_melspectrogram(input_data=y,input_sampling_rate=sr,output_file_path=output_file)
            if output_file is not None:
                saved_output_file_names.append(output_file)
                num_images_generated += 1
                
            current_chunk_time_start += chunk_duration
            current_chunk_time_end += chunk_duration

        return saved_output_file_names

    except Exception as e:
        print(f"Error while generating and saving melspectrogram stream for {input_file_path}: {e}")
        return None

"""
    @method get_classified_genre
        Get the classified genre from a HouseX-trained model
    @param input_spectrogram_path: Path to the input spectrogram image
    @param model_path: Path to the `.pth` file containing the genre classification model
    @note The model will be loaded with the `torch.load(PATH)` moniker.
"""
def get_classified_genre(input_spectrogram_path:str, model_path:str) -> Optional[str]:
    try:
        global DEVICE, IDX_TO_LABEL_MAPPING

        # Transform the input image into a torch tensor @ref: https://www.projectpro.io/recipes/convert-image-tensor-pytorch
        transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(input_spectrogram_path).convert('RGB')
        transformed_img = transform(img=img)
        x = torch.Tensor(transformed_img)
        x = x.to(DEVICE)

        # Load model
        model = torch.load(model_path)
        model.to(DEVICE)
        model.eval()

        # Compute outputs
        with torch.no_grad():
            outputs = model(x.unsqueeze(0))
            y = torch.softmax(outputs, dim = 1).detach().cpu()
            selected_index = int(torch.argmax(y).item())

        selected_genre = IDX_TO_LABEL_MAPPING[selected_index]
        return selected_genre
    except Exception as e:
        print(f"Error while classifying genre of {input_spectrogram_path}: {e}")
        return None