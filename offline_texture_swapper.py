#https://github.com/S-aiueo32/srntt-pytorch/blob/master/offline_texture_swapping.py

import os
import argparse
import numpy as np

from data_loader import SwappingDataset

import torch
from torch.utils.data import DataLoader

from model.vgg import VGG
from swapper import Swapper

TARGET_LAYERS = ['relu3_1', 'relu2_1', 'relu1_1']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', default=3)
    parser.add_argument('--stride', default=1)
    
    args, unknown = parser.parse_known_args()
    

    model = VGG(model_type='vgg19').to(device)
    swapper = Swapper(args.patch_size, args.stride).to(device)
    
    dataset = SwappingDataset('CUFED5')
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    
    save_path = os.path.join(dataset.path, 'texture')
    os.makedirs(save_path, exist_ok=True)

    print("Offline Swapping...")
    for i, (data) in enumerate(data_loader):
        input_imgs, lowRef_imgs, highRef_imgs, filenames = data 
        input_imgs = input_imgs.to(device)
        lowRef_imgs = lowRef_imgs.to(device)
        highRef_imgs = highRef_imgs.to(device)
        

        map_in = model(input_imgs, TARGET_LAYERS)
        map_ref = model(highRef_imgs, TARGET_LAYERS)
        map_ref_blur = model(lowRef_imgs, TARGET_LAYERS)

        maps, weights, correspondences = swapper(map_in, map_ref, map_ref_blur)
        
        for i, filename in enumerate(filenames):
            np.savez_compressed(os.path.join(save_path, f'{filename}.npz'),
                relu1_1=maps['relu1_1'],
                relu2_1=maps['relu2_1'],
                relu3_1=maps['relu3_1'],
                weights=weights,
                correspondences=correspondences)
            
    print("Done")
            
