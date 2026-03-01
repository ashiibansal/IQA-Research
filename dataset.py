import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class DegradationDataset(Dataset):
    """
    Custom Dataset class to load our ruined images and their CSV recipes.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Define the exact Min and Max bounds we used in our generation script
        self.bounds = {
            'blur_sigma': (0.0, 3.0),
            'noise_std': (0.0, 25.0),
            'jpeg_quality': (10.0, 100.0),
            'gamma_exposure': (0.5, 1.5)
        }

    def __len__(self):
        return len(self.data_frame)

    def normalize(self, value, col_name):
        """Applies Min-Max normalization to squish values between 0.0 and 1.0"""
        min_val, max_val = self.bounds[col_name]
        return (value - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        # 1. Locate and Load the Image
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        # 2. Extract the Raw Labels from the CSV row
        raw_blur = self.data_frame.iloc[idx, 1]
        raw_noise = self.data_frame.iloc[idx, 2]
        raw_jpeg = self.data_frame.iloc[idx, 3]
        raw_gamma = self.data_frame.iloc[idx, 4]

        # 3. Normalize the Labels
        norm_blur = self.normalize(raw_blur, 'blur_sigma')
        norm_noise = self.normalize(raw_noise, 'noise_std')
        norm_jpeg = self.normalize(raw_jpeg, 'jpeg_quality')
        norm_gamma = self.normalize(raw_gamma, 'gamma_exposure')

        # 4. Convert Labels to a PyTorch Tensor
        labels = torch.tensor([norm_blur, norm_noise, norm_jpeg, norm_gamma], dtype=torch.float32)

        # 5. Apply Vision Transformations
        if self.transform:
            image = self.transform(image)

        return image, labels