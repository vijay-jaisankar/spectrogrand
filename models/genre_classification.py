"""
    Contains the model definition of the finetuned Resnet-101 model to detect the genre of a music piece as described by its spectrogram (housex-task)
"""

import torch
import torch.nn as nn 
import torch.nn.functional as f
import torchvision.models as models

class GenreNet(nn.Module):
    def __init__(self, dropout_rate:float = 0.50, num_output_classes:int = 4):
        super().__init__()

        # Set instance variables
        self.dropout_rate = dropout_rate
        self.num_output_classes = num_output_classes

        # Set the current device for tensor calculations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Baseline: Resnet 101
        self.backbone = models.resnet101(weights="IMAGENET1K_V1")

        self.model = nn.Sequential(
            self.backbone,
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1000, self.num_output_classes)
        )

    def forward(self, x):
        # Inference
        x = x.to(self.device)
        x = self.model(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    MODEL_ARGS = {
        "num_output_classes" : 4,
        "dropout_rate" : 0.50
    }

    model = GenreNet(**MODEL_ARGS).to(device)