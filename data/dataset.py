import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    def __init__(self, df, image_dir, params, transform=None) -> None:
        super().__init__()
        self.df = df
        self.image_dir = image_dir
        self.params = params
        self.transform = transform
        
    def __len__(self) -> int:
        # return size of dataset
        return self.df.shape[0]/2
    
    def __getitem__(self, idx:int):
        df = self.df[self.df.CollectionId==idx]
        label = df.Genus[0]
        image = torch.stack([self.get_image(os.path.join(self.image_dir, os.path.splitext(file)+".jpg")) for file in df.FileName])
        return image, label

    def get_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, self.params.shape)
        image /= 255.0

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image