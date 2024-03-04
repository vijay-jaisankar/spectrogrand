"""
    Collection of helper functions pertaining to the spectrogram domain of spectrogrand
"""
from typing import Optional, List
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

torch.random.manual_seed(42)

import tensorflow as tf
import tensorflow_hub as hub


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF_DEVICE = "/gpu:0" if torch.cuda.is_available() else "/cpu"

from diffusers import DiffusionPipeline

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
sdxl_pipeline.to(TORCH_DEVICE)

# Load the VILA pipeline
vila_model = hub.load('https://tfhub.dev/google/vila/image/1')
vila_predict_fn = vila_model.signatures['serving_default']

# Load the surprise estimation pipeline
class CreativeNet(nn.Module):
    def __init__(self, train_baseline_classifier = False, num_output_classes = 2, dropout_rate = 0.20):
        super().__init__()
        
        # Set instance variables
        self.train_baseline_classifier = train_baseline_classifier
        self.num_outuput_classes = num_output_classes
        self.dropout_rate = dropout_rate
        
        # Set the current device for tensor calculations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    @param output_dir: Path to the parent directory where the contiguous melspectrogram images are to be stored
    @param num_inference_steps: Number of diffusion steps SDXL will take to generate the image (@note The higher this number, the longer the pipeline will take while maintaining higher-quality outputs) (default: 50)
"""
def generate_and_save_image_stream(genre_name:str, output_dir:str, num_inference_steps:int=50) -> Optional[List[str]]:
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
                prompt = f"{colour} these colored album cover for music that {word}"
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

        img = Image.open(input_image_path)
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