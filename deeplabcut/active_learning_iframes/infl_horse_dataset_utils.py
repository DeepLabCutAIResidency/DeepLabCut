# imports

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision import transforms
# from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from PIL import Image
from tqdm import tqdm


#####################################################################################
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, df_groundtruth, img_dir, transform=None): # img_dir: parent dir to labeled-data
        self.df_groundtruth = df_groundtruth #pd.read_hdf(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df_groundtruth)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 
                                self.df_groundtruth.index[idx])
        image = Image.open(img_path) 
        if self.transform:
            image = self.transform(image)
        return image #, label



#####################################################################################
# def rank_idcs_by_influence(idcs_to_rank,
#                            idcs_for_reference,
#                            full_feature_matrix,
#                            device):
#     ### Create dataloader
#     params = {'batch_size': 64,
#           'shuffle': False,
#           'num_workers': 6}
#     df_dataloader = torch.utils.data.DataLoader(df_dataset, 
#                                                 **params)

#     ### Compute feature vectors
#     list_feature_tensors_per_sample = []
#     with torch.no_grad():
#         for data_batch in tqdm(df_dataloader):            
#             out = alexnet_feature_extractor(data_batch.to(device))
#             list_feature_tensors_per_sample.append(out[alexnet_node_output_str]) 

#     # feature_tensors = torch.cat(alexnet_feature_extractor)  
#     return torch.cat(alexnet_feature_extractor)  
                    
