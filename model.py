import torch.nn as nn
from torchvision import models

class RestorationParameterPredictor(nn.Module):
    def __init__(self):
        super(RestorationParameterPredictor, self).__init__()
        
        # 1. Load the pre-trained ResNet-50 backbone
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        # 2. Extract the number of input features going into the final layer
        num_ftrs = self.backbone.fc.in_features
        
        # 3. Replace the classification head with our 4-parameter regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),          
            nn.Linear(512, 4)  # Outputs: Blur, Noise, JPEG, Gamma
        )

    def forward(self, x):
        return self.backbone(x)