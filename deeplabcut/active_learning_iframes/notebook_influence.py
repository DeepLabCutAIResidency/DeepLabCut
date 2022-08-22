'''
Exploring ranking new data by influence

from:
https://pytorch.org/hub/pytorch_vision_alexnet/
https://stackoverflow.com/questions/51501828/how-to-extract-fc7-features-from-alexnet-in-pytorch-as-numpy-array
'''

#########################################################
# %%
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
# %% Define dataset class
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, df_groundtruth, img_dir, transform=None): # img_dir: parent dir to labeled-data
        self.df_groundtruth = df_groundtruth #pd.read_hdf(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.df_groundtruth)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 
                                self.df_groundtruth.index[idx])
        image = Image.open(img_path) #read_image(img_path)
        # label = self.df_groundtruth.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image #, label

##################################################################################
# %%
# Input data
reference_dir_path = '/home/sofia/datasets/Horse10_AL_unif_OH'
path_to_pickle_w_base_idcs = os.path.join(reference_dir_path,
                                          'horses_AL_OH_train_test_idcs_split.pkl') #TODO these should probably be a unique file, not copies over each AL approach
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!
gpu_to_use = None # with gpu faster but I get out of memory error

alexnet_node_output_str = 'classifier.2'
################################################
# %% Select GPU or CPU
use_cuda = torch.cuda.is_available()

if use_cuda and gpu_to_use != None:
    device = torch.device("cuda:{}".format(gpu_to_use))
else:
    device = 'cpu'
torch.backends.cudnn.benchmark = True

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20'---potentiallly to fix OOM error?
#############################################
# %%
## Preprocess data as expected by the model
# All pre-trained models expect input images normalized in the same way, 
# i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
# H and W are expected to be at least 224. The images have to be loaded 
# in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] 
# and std = [0.229, 0.224, 0.225].
# 
# pytorch transformations accept PIL Image, Tensor Image or batch of Tensor Images as input.
# Batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch.   
preprocess = transforms.Compose([transforms.Resize(224), # transforms.Resize(256),  transforms.CenterCrop(224), # transforms.Resize(244), # before: 
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std=[0.229, 0.224, 0.225]),])


dataset = CustomImageDataset(pd.read_hdf(path_to_h5_file), #path_to_h5_file,
                             os.path.join(reference_dir_path),
                             transform=preprocess)
dataset[0].shape
###############################################################
# %% # Get feature extractor from AlexNet pretrained model

# fetch model
alexnet_model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
# nodes, _ = get_graph_node_names(alexnet_model)
# print(nodes)

# create a feature extractor
alexnet_feature_extractor = create_feature_extractor(
	alexnet_model, return_nodes=[alexnet_node_output_str]) # see print(alexnet_model)

# do I need eval()? 
alexnet_feature_extractor.eval()
alexnet_feature_extractor.to(device)

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')


# ################################################    
# # %% Try with one tensor
# filename = '/home/sofia/datasets/Horse10_AL_uncert_OH/labeled-data/BrownHorseinShadow/0052.png'
# input_image = Image.open(filename) # range from 0 to 255 # (288, 162)

# input_tensor = preprocess(input_image) # torch.Size([3, 224, 398])
# input_batch = input_tensor.unsqueeze(0) # torch.Size([1, 3, 224, 398])
# out = alexnet_feature_extractor(input_batch) # torch.Size([1, 4096])

# ################################################    
# # %% Try looping over every tensor in dataset
# list_features_per_img = []
# co=1
# for input_tensor in dataset:
#     input_batch = input_tensor.unsqueeze(0) # torch.Size([1, 3, 224, 398])
#     out = alexnet_feature_extractor(input_batch) 
#     list_features_per_img.append(out)
#     print(co)
#     co+=1

################################################        
# %% Try using dataloader
# `out` will be a dict of Tensors, each representing a feature map
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)  #torch.Size([1, 3, 224, 224])
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20'
params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}
dataset_generator = torch.utils.data.DataLoader(dataset, **params)


list_feature_tensors_per_img = []
# co=1
for data_batch in tqdm(dataset_generator):
    # print(co)
    # co+=1
    # print(data_sample.unsqueeze(0).shape) #torch.Size([64, 3, 244, 433])
    with torch.no_grad():
        out = alexnet_feature_extractor(data_batch.to(device))
    list_feature_tensors_per_img.append(out[alexnet_node_output_str]) 

feature_tensors = torch.cat(list_feature_tensors_per_img)

#################################################################################
# %% Compute cosine sim between numpy arrays 
feature_arrays = feature_tensors.cpu().numpy() # nrows= observations, ncols=dimensions of space

# OJO! cosine distance between vectors u and v computed as 1 - cos(angle between u,v)
dist_array = cdist(feature_arrays,
                    feature_arrays,
                    'cosine')

plt.matshow(dist_array)
plt.show()
## %%timeit  
# dist_array = cdist(feature_arrays,
                    # feature_arrays,
                    # 'cosine')
#################################################################################
# %% Compute cosine sim between torch tensors -2 
# https://discuss.pytorch.org/t/compute-the-row-wise-distance-of-two-matrix/4541
cosi = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

cos_dist_array = torch.empty((feature_tensors.shape[0],
                              feature_tensors.shape[0]))
for i in range(cos_dist_array.shape[0]):           
    for j in range(cos_dist_array.shape[1]):                         
        cos_dist_array[i,j] = 1.0 - cosi(feature_tensors[i,:].unsqueeze(0), 
                                        feature_tensors[j,:].unsqueeze(0)) #.numpy()
        # cos_sim.shape

cos_dist_np = cos_dist_array.numpy()
plt.matshow(cos_dist_np)
plt.show()

# np.max(abs(cos_dist_np-dist_array))---> 3e07                


#############################################################
# %% Compute cosine similarity between torch tensors  

# Compute from Euclidean distance using law of cosines (check if ok?)
# Euclidean distance (non-squared!)
# not sure if compute mode refers to this: https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf (why a choice?)
eucl_pairwise_dist = torch.cdist(feature_tensors,
                                 feature_tensors,
                                 p=2,
                                 compute_mode='use_mm_for_euclid_dist_if_necessary') #donot_use_mm_for_euclid_dist

row_vectors_norms = torch.linalg.vector_norm(feature_tensors, ord=2, dim=1)
col_vectors_norms = torch.linalg.vector_norm(feature_tensors, ord=2, dim=1)
grid_rows, grid_cols = torch.meshgrid(row_vectors_norms,
                                      col_vectors_norms, indexing='ij')

                        
cos_sim = 1 - (eucl_pairwise_dist**2 - grid_rows**2 - grid_cols**2)/\
              (-2*grid_rows*grid_cols) # add max(x,eps) to denominator?

cos_sim_np = cos_sim.cpu().numpy()
plt.matshow(cos_sim_np)
plt.show()

#############################################################
# %% Compute cosine similarity between torch tensors  

# Compute from Euclidean distance using law of cosines (check if ok?)
# Euclidean distance (non-squared!)
# not sure if compute mode refers to this: https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf (why a choice?)
eucl_pairwise_dist = torch.cdist(feature_tensors,
                                 feature_tensors,
                                 p=2,
                                 compute_mode='use_mm_for_euclid_dist_if_necessary') #donot_use_mm_for_euclid_dist

sq_eucl_pairwise_dist = eucl_pairwise_dist**2

row_vectors_norms = torch.linalg.vector_norm(feature_tensors, ord=2, dim=1)
col_vectors_norms = torch.linalg.vector_norm(feature_tensors, ord=2, dim=1)

cos_sim = torch.empty((feature_tensors.shape[0],
                              feature_tensors.shape[0]))
for i in range(cos_dist_array.shape[0]):           
    for j in range(cos_dist_array.shape[1]):                         
        cos_sim[i,j] = \
             1 - (sq_eucl_pairwise_dist[i,j] - row_vectors_norms[i]**2 - col_vectors_norms[j]**2)/\
                 (-2*row_vectors_norms[i]*col_vectors_norms[j]) # add max(x,eps) to denominator?

cos_sim_np = cos_sim.cpu().numpy()
plt.matshow(cos_sim_np)
plt.show()

# np.max(abs(cos_dist_np-dist_array))
###################################################################
# %% Checking faster ways to compute cosine similarity....
# https://github.com/pytorch/pytorch/issues/48306 ---Memory allocation error
# also here: # https://github.com/pytorch/pytorch/issues/11202 
cosine_sim = nn.functional.cosine_similarity(feature_tensors[:,None,:], 
                                             feature_tensors[None,:,:],
                                             dim=1) 

##################################################################################
# %% Compute cosine similarity between torch tensors -1
# https://github.com/pytorch/pytorch/issues/11202

# def cosine_pairwise(x):
#     x = x.permute((1, 2, 0))
#     cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
#     cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
#     return cos_sim_pairwise

# pairwise_sim = cosine_pairwise(feature_tensors.unsqueeze(0))


# %%
