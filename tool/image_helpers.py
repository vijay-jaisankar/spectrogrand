"""
    Collection of helper functions pertaining to the spectrogram domain of spectrogrand
"""
from typing import Optional, List
from PIL import Image
from io import BytesIO
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

torch.random.manual_seed(42)

import tensorflow as tf
import tensorflow_hub as hub


TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TF_DEVICE = "/gpu:0" if torch.cuda.is_available() else "/cpu"
# Use another GPU core for SDXL if possible
SD_DEVICE = "cpu"
if int(torch.cuda.device_count()) == 1:
    SD_DEVICE = "cuda:0"
else:
    SD_DEVICE = "cuda:1"

from diffusers import DiffusionPipeline

# @note These defaults can be changed based on the user's preferences
GENRE_COLOUR_MAPPING = {
    'future house' : ["blue", "red"],
    'bass house' : ["black", "purple"],
    'progressive house' : ["orange", "yellow"],
    'melodic house' : ["green", "blue"]
}

GENRE_WORD_MAPPING = {
    'future house' : ["unveils", "surprises"],
    'bass house' : ["ascends", "explodes"],
    'progressive house' : ["balances", "hypnotizes"],
    'melodic house' : ["blends", "stuns"]
}


# Load the Stabke Duffusion XL pipeline
sdxl_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
sdxl_pipeline.to(SD_DEVICE)

# Load the VILA pipeline
vila_model = hub.load('https://tfhub.dev/google/vila/image/1')
vila_predict_fn = vila_model.signatures['serving_default']

# Load the Magenta pipeline
magenta_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load the surprise estimation pipeline
class CreativeNet(nn.Module):
    def __init__(self, train_baseline_classifier = False, num_output_classes = 2, dropout_rate = 0.20):
        super().__init__()
        
        # Set instance variables
        self.train_baseline_classifier = train_baseline_classifier
        self.num_outuput_classes = num_output_classes
        self.dropout_rate = dropout_rate
        
        # Set the current device for tensor calculations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Baseline: MobileNet V3 small
        self.baseline = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Freeze the parameters of the base model (including but not limited to the last layers)
        for param in self.baseline.parameters():
            param.requires_grad = False
        
        if self.train_baseline_classifier:
            for param in self.baseline.classifier.parameters():
                param.requires_grad = True
                
        # Fully-connected block
        self.fc1 = nn.Linear(1000, 128)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(32, self.num_outuput_classes)
        
    def forward(self, x):
        # Baseline
        x = x.to(self.device)
        x = self.baseline(x)
        
        # FC Block
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x))
        x = torch.sigmoid(x)
        return x
    
surprise_model_args = {
        "train_baseline_classifier" : False, 
        "num_output_classes" : 2,
        "dropout_rate" : 0.35
    }
surprise_model = CreativeNet(**surprise_model_args).to(TORCH_DEVICE)

"""
    @method generate_and_save_image
        Generate and save an image for a text prompt 
    @param prompt: Textual prompt containg the specifics of the image to be generated
    @param output_file_path: Path to the output file where the generated image is to be stored
    @param num_inference_steps: Number of diffusion steps SDXL will take to generate the image (@note The higher this number, the longer the pipeline will take while maintaining higher-quality outputs) (default: 50)
"""
def generate_and_save_image(prompt:str, output_file_path:str, num_inference_steps:int=50) -> Optional[str]:
    try:
        global sdxl_pipeline
        # Generate image
        images = sdxl_pipeline(prompt=prompt, num_inference_steps=num_inference_steps)
        img = images[0][0]
        # Save image
        img.save(output_file_path)
        return output_file_path
    except Exception as e:
        print(f"Error while generating image: {e}")
        return None
    
"""
    @method generate_and_save_image_stream
        Generate and save genre-driven candidate album covers
        @note To change the genre-mapping configs, rewrite GENRE_COLOR_MAPPING and/or GENRE_WORD_MAPPING before calling this function.
    @param genre_name: Name of the genre, as it appears in the keys of GENRE_COLOR_MAPPING and GENRE_WORD_MAPPING
    @param topic: Topic of the music piece
    @param output_dir: Path to the parent directory where the contiguous melspectrogram images are to be stored
    @param num_inference_steps: Number of diffusion steps SDXL will take to generate the image (@note The higher this number, the longer the pipeline will take while maintaining higher-quality outputs) (default: 50)
"""
def generate_and_save_image_stream(genre_name:str, topic:str, output_dir:str, num_inference_steps:int=50) -> Optional[List[str]]:
    try:
        global GENRE_COLOUR_MAPPING, GENRE_WORD_MAPPING
        # Keep running count of the current time index and number of images generated
        num_images_generated = 0
        saved_output_file_names = []

        # Generate and save an image using the GridSearch heuristic
        for colour in GENRE_COLOUR_MAPPING[genre_name]:
            for word in GENRE_WORD_MAPPING[genre_name]:
                output_file = f"{output_dir}/sdxl{num_images_generated}.png"
                # Construct the prompt
                prompt = f"{colour} colored album cover for music about {topic} that {word}"
                output_file = generate_and_save_image(prompt=prompt,output_file_path=output_file,num_inference_steps=num_inference_steps)
                if output_file is not None:
                    saved_output_file_names.append(output_file)
                    num_images_generated += 1
        
        return saved_output_file_names
    except Exception as e:
        print(f"Error while generating and saving image stream: {e}")
        return None

"""
    @method get_vila_score
        Score an image for its aesthetic qualities using the VILA model
    @param input_image_path: Path to the input image
"""
def get_vila_score(input_image_path:str) -> Optional[float]:
    try:
        global TF_DEVICE   , vila_predict_fn, vila_model 
        # Load image
        img = Image.open(input_image_path)
        # Convert image to Bytes array @ref https://stackoverflow.com/a/33117447
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format=img.format)
        img_byte_arr = img_byte_arr.getvalue()
        # Get predictions
        with tf.device(TF_DEVICE):
            prediction = vila_predict_fn(tf.constant(img_byte_arr))
            return float(prediction['predictions'][0][0])
    except Exception as e:
        print(f"Error while calculating VILA score for {input_image_path}: {e}")
        return None

"""
    @method get_surprise_score
        Get the surprise coefficient from a MUMU-trained model
    @param input_image_path: Path to the input SDXL image
    @param model_path: Path to the `.pt` file containing the surprise estimation model
    @note The model will be loaded with the `model.load_state_dict(torch.load(PATH))` moniker.
"""
def get_surprise_score(input_image_path:str, model_path:str) -> Optional[str]:
    try:
        global TORCH_DEVICE, surprise_model   

        # Transform the input image into a torch tensor @ref: https://www.projectpro.io/recipes/convert-image-tensor-pytorch
        transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
        ])

        img = Image.open(input_image_path).convert("RGB")
        transformed_img = transform(img=img)
        x = torch.Tensor(transformed_img)
        x = x.to(TORCH_DEVICE)

        # Load model
        surprise_model.load_state_dict(torch.load(model_path))
        surprise_model.to(TORCH_DEVICE)
        surprise_model.eval()

        # Compute outputs
        with torch.no_grad():
            outputs = surprise_model(x.unsqueeze(0))
            y = torch.softmax(outputs, dim = 1).detach().cpu()
            selected_score = float(y[0][1].item()) # Order of scores: ai, human
        return selected_score
    except Exception as e:
        print(f"Error while classifying genre of {input_image_path}: {e}")
        return None
    
"""
    @method neural_style_transfer_vanilla
        Perform fast neural style transfer using the Magenta model
        @note ref: https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/tensorFlow1/variations/256/versions/2?tfhub-redirect=true
    @param content_image_path: Path to the input image that serves as the content image
    @param style_image_path: Path to the input image that serves as the style image
    @param output_file_path: Path to which the style transfer output is to be stored 
"""
def neural_style_transfer_vanilla(content_image_path:str, style_image_path:str, output_file_path:str) -> Optional[str]:
    # Nested function to load a PIL image as a TF tensor
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    
    # Nested function to convert TF tensor back to PIL image
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    try:
        global magenta_model
        content_image_tensor = load_img(content_image_path)
        style_image_tensor = load_img(style_image_path)
        stylised_image_tensor = magenta_model(tf.constant(content_image_tensor), tf.constant(style_image_tensor))[0]
        stylised_image_pil = tensor_to_image(stylised_image_tensor)
        stylised_image_pil.save(output_file_path)
        return output_file_path
    except Exception as e:
        print(f"Error while performing vanilla neural style transfer: {e}")
        return None
    
"""
    @method neural_style_transfer_vanilla_stream
        Perform OvA (one-vs-all) Neural style transfer with a single content/style image and a directory of style/content images
    @param single_image_path: Path to the single input image that serves as the content image
    @param stream_image_dir: Path to the directory containing the input images that serves as the content/style images
    @param ova_mode: Mode of neural style transfer. This parameter takes two values: "style" for single style image and a stream of content images; and "content" for a single content image and a stream of style images
    @param output_dir: Path to the parent directory where the contiguous melspectrogram images are to be stored
"""
def neural_style_transfer_vanilla_stream(single_image_path:str, stream_image_dir:str, ova_mode:str, output_dir:str) -> Optional[List[str]]:
    try:
        # Maintain a list of stream image filenames and count of generated images
        stream_image_filenames = glob(f"{stream_image_dir}/*")
        assert len(stream_image_filenames) > 0

        # Keep running count of the current time index and number of images generated
        num_images_generated = 0
        saved_output_file_names = []

        for stream_image in tqdm(stream_image_filenames):
            output_file = f"{output_dir}/nst_{num_images_generated}.png"
            # Check OvA mode
            if ova_mode == "style":
                output_file = neural_style_transfer_vanilla(content_image_path=stream_image,style_image_path=single_image_path,output_file_path=output_file)
                if output_file is not None:
                    saved_output_file_names.append(output_file)
                    num_images_generated += 1

            if ova_mode == "content":
                output_file = neural_style_transfer_vanilla(content_image_path=single_image_path,style_image_path=stream_image,output_file_path=output_file)
                if output_file is not None:
                    saved_output_file_names.append(output_file)
                    num_images_generated += 1

        return saved_output_file_names

    except Exception as e:
        print(f"Error while generating and saving neural style transfer image stream: {e}")
        return None
    