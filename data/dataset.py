import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BagDataset(Dataset):
    def __init__(self, df, params, transform=None) -> None:
        super().__init__()
        
    def __len__(self) -> int:
        # return size of dataset
        return len(self.images)
    
    def __getitem__(self, index:int):
        return 