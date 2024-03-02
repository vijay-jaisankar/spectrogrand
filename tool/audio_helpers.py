"""
    Collection of helper functions pertaining to the audio domain of spectrogrand
"""
from diffusers import AudioLDM2Pipeline
from typing import Optional
import scipy
import numpy as np
import librosa
import pickle

import torch
torch.random_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import ClapModel, ClapProcessor


# Load the AudioLDM pipeline
audio_ldm_pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16)
audio_ldm_pipeline.to(DEVICE)

# Load the CLAP pipeline
clap_model = ClapModel.from_pretrained("laion/clap-hstat-fused")
clap_model.to(DEVICE)
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

"""
    @method create_and_save_audio_file
        Use the `audioldm2-music` model to generate synthetic audio conditioned on inputs
    @param text_prompt: Descriptor of the audio to be generated (@note The user is requested to be as verbose as possible)
    @param output_file_path: Path to which the generated audio is to saved
    @param num_inference_steps: Number of inference steps for the audioldm2-music model (@note The higher this value, the longer the po)

"""
def create_and_save_audio_file(text_prompt:str, output_file_path:str, num_inference_steps:int=500, audio_length:float=10.0, negative_prompt = "low quality, monotonous, boring") -> Optional[str]:
    try:
        global audio_ldm_pipeline
        # Generate audio
        audio = audio_ldm_pipeline(
            text_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=float(audio_length),
            num_waveforms_per_prompt=2
        ).audios[0]
        # Save audio
        scipy.io.wavfile.write(output_file_path, rate=16000, data=audio)
        return output_file_path
    except Exception as e:
        print(f"Error while generating and saving audio: {e}")
        return None

"""
    @method load_wav_file
        Load a wav file from a given path
    @param input_file_path: Path containing the input audio file
"""
def load_wav_file(input_file_path:str):
    try:
        sr, data = scipy.io.wavfile.read(input_file_path)
        return sr, data
    except Exception as e:
        print(f"Error while reading {input_file_path}: {e}")
        return None, None
    
"""
    @method resample_audio_data
        Resample audio data to a target sampling rate
    @param origin_data: Numpy array containing the audio data (@note Intended inputs stem from the `load_wav_file` method)
    @param origin_sampling_rate: Sampling rate of the input audio (@note Intended inputs stem from the `load_wav_file` method)
    @param new_sampling_rate: Desired sampling rate (default: 48000)
"""
def resample_audio_data(origin_data:np.ndarray, origin_sampling_rate:int, new_sampling_rate:int=48000) -> Optional[np.ndarray]:
    try:
        origin_type = origin_data.dtype
        resampled_data = librosa.resample(origin_data.T.astype('float'), orig_sr = origin_sampling_rate, target_sr = new_sampling_rate) 
        resampled_data = librosa.to_mono(resampled_data)        
        resampled_data = resampled_data.T.astype(origin_type)
        data_np = np.array(resampled_data)
        return data_np
    except Exception as e:
        print(f"Error while resampling audio data: {e}")
        return None

"""
    @method compute_clap_embeddings
        Compute CLAP embeddings for an input audio file
    @input input_file_path: Path to the input audio file
"""
def compute_clap_embeddings(input_file_path:str) -> Optional[torch.Tensor]:
    try:
        global clap_processor, clap_model, DEVICE
        # Load audio and resample to 48000 Hz
        sr, origin_data = load_wav_file(input_file_path=input_file_path)
        origin_data_resampled = resample_audio_data(origin_data=origin_data, origin_sampling_rate=sr, new_sampling_rate=48000)
        # Get CLAP outputs
        clap_inputs = clap_processor(audios=origin_data_resampled, sampling_rate=48000, return_tensors="pt").to(DEVICE)
        clap_outputs = clap_model.get_audio_features(**clap_inputs)
        audio_embeds = clap_outputs[0].detach().cpu()
        return audio_embeds
    except Exception as e:
        print(f"Error while computing CLAP embeddings for {input_file_path}: {e}")
        return None
    
"""
    @method compute_clap_similarity
        Compute CLAP similarity for an input audio file with respect to a saved ground truth mapping of embeddings
        @input input_file_path: Path to the input audio file
        @input ground_truth_dict_path: Path to the mapping .pkl file 
        @note The ground truth mapping should be a .pkl file with the following schema:
            {
                "genre_name" : [list_of_clap_embeddings],
                ...
            }
        @input filter_genre: Genre name to compute from. If values are to be aggregated across the entire search space, this value should be left as `None`. (default: None)
"""
def compute_clap_similarity(input_file_path:str, ground_truth_dict_path:str, filter_genre:Optional[str]=None) -> Optional[float]:
    try:
        # Load the embeddings from the ground truth mapping and set the search space
        with open(ground_truth_dict_path, "rb") as f:
            data = pickle.load(f)
        input_search_space_embeds = []
        if filter_genre is not None:
            input_search_space_embeds = data[filter_genre]
        else:
            for _k in data:
                input_search_space_embeds.extend(data[_k])
        assert len(input_search_space_embeds) >= 1

        # Compute CLAP embeddings for the input file
        source_embed = compute_clap_embeddings(input_file_path=input_file_path)

        # Keep track of running dot product scores
        running_score = 0.0
        for target_embed in input_search_space_embeds:
            z = source_embed@target_embed.T
            running_score += float(z.detach().cpu())

        # Return the average dot product score
        return (running_score)/float(len(input_search_space_embeds))
    except Exception as e:
        print(f"Error while computing CLAP similarity score for {input_file_path}: {e}")
        return None