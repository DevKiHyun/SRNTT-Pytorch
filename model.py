import os
import cv2 
import glob
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, inputs_list, ref_list):
        """
        Args:
            Dataset api for Image data.
            inputs_list (list): Path to the csv file with annotations.
            labels_list (list): Directory with all the images.
        """
        self.inputs_list = inputs_list
        self.ref_list = ref_list

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        get_image = lambda path : cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        input_path = self.inputs_list[index]
        ref_path   = self.ref_list[index]
        
        input_img = get_image(input_path)
        ref_img   = get_image(ref_path)

        return [input_img, ref_img]