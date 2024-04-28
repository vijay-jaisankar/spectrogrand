"""
    Collection of helper functions pertaining to the audio domain of spectrogrand
"""
from diffusers import AudioLDM2Pipeline
from typing import Optional, List
import scipy
import numpy as np
import librosa
import pickle
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

import torch
torch.random.manual_seed(42)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import ClapModel, ClapProcessor


# Load the AudioLDM pipeline
audio_ldm_pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music")
audio_ldm_pipeline.to(DEVICE)

# Load the CLAP pipeline
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
clap_model.to(DEVICE)
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

"""
    @method create_and_save_audio_file
        Use the `audioldm2-music` model to generate synthetic audio conditioned on inputs
    @param text_prompt: Descriptor of the audio to be generated (@note The user is requested to be as verbose as possible)
    @param output_file_path: Path to which the generated audio is to saved
    @param num_inference_steps: Number of inference steps for the audioldm2-music model (@note The higher this value, the longer the time for execution of this step)
    @param audio_length: Length of the audio piece (in s) (default: 10.0)
    @param negative_prompt: Negative prompt to be passed to the audioldm2-music model (default: 'low quality, monotonous, boring')
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
            # Convert `filter_genre` into underscore format if required
            if "_" not in filter_genre: # eg: 'bass house'
                filter_genre = filter_genre.replace(" ","_")
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
    
"""
    @method load_wav_chunk
        Load a time-defined chunk of audio from a wav file
    @param input_file_path: Path to the input wav file
    @param chunk_offset: Timestamp (in s) from which the chunk starts
    @param chunk_duration: Duration of the returned chunk (in s) (default: 0.1)
"""
def load_wav_chunk(input_file_path:str, chunk_offset:float, chunk_duration:float=0.1):
    try:
        y, sr = librosa.load(path=input_file_path, offset=float(chunk_offset), duration=float(chunk_duration), sr=None)
        return sr, y
    except Exception as e:
        print(f"Error while loading chunk from {input_file_path}: {e}")
        return None, None
    
"""
    @method create_and_save_audio_file_stream
        Use the `audioldm2-music` model to generate synthetic audio conditioned on inputs
    @param topic: Topic of the audio files to be generated
    @param output_dir: Directory to which the generated audio files are to saved
    @param num_inference_steps_list: List containing the number of inference steps (default: [500, 750])
    @param audio_length: Length of the audio piece (in s) (default: 10.0)
    @param negative_prompt: Negative prompt to be passed to the audioldm2-music model (default: 'low quality, monotonous, boring')
"""
def create_and_save_audio_file_stream(topic:str, output_dir:str, num_inference_steps_list:list=[500, 750], audio_length:float=10.0, negative_prompt = "low quality, monotonous, boring") -> Optional[List[str]]:
    try:
        global audio_ldm_pipeline
        # Generate audio
        # Keep running count of the current time index and number of images generated
        num_audios_generated = 0
        saved_output_file_names = []
        
        # Construct the text prompt
        text_prompt = f"Jumpy electronic house music for {topic}"

        for num_infer in num_inference_steps_list:
            output_file = f"{output_dir}/audio{num_audios_generated}.wav"
            output_file = create_and_save_audio_file(text_prompt=text_prompt, output_file_path=output_file, num_inference_steps=num_infer, audio_length=audio_length, negative_prompt=negative_prompt)
            if output_file is not None:
                saved_output_file_names.append(output_file)
                num_audios_generated += 1

        return saved_output_file_names
    except Exception as e:
        print(f"Error while generating and saving audio stream: {e}")
        return None
    
"""
    @method get_danceability_score
        Use Essentia to score an audio track on its danceability
    @input input_file_path: Path to the input audio file
    @input embedding_model_path: Path to the essentia encoder model (@note To ensure compatibility, this should be a `.pb` file)
    @input danceability_model_path: Path to the essentia danceability computation model (@note To ensure compatibility, this should be a `.pb` file)
"""
def get_danceability_score(input_file_path:str, embedding_model_path:str, danceability_model_path:str) -> Optional[float]:
    try:
        # Load audio and get embeddings
        audio = MonoLoader(filename=input_file_path, sampleRate=16000, resampleQuality=4)()
        embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embedding_model_path, output="PartitionedCall:1")
        embeddings = embedding_model(audio)

        # Load model and get predictions
        model = TensorflowPredict2D(graphFilename=danceability_model_path, output="model/Softmax")
        predictions = model(embeddings)
        mean_danceability_score = np.mean(predictions[:,0])
        return mean_danceability_score
    except Exception as e:
        print(f"Error while computing danceability score for {input_file_path}: {e}")
        return None