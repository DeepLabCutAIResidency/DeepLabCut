'''

Script to rank active learning train idcs by influence metric 
(as defined in Liu and Ferrari, Active learning for human pose estimation)
https://pytorch.org/hub/pytorch_vision_alexnet/

'''
#####################################################
# %%
import os
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# import timeit

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor # get_graph_node_names

from PIL import Image
from tqdm import tqdm


from deeplabcut.active_learning_iframes.infl_horse_dataset_utils import CustomImageDataset, pairwise_cosine_distance, pairwise_cosine_distance_two_inputs
#####################################################
# %%
# Inputs
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_test_idcs_split.pkl' 
reference_dir_path = '/home/sofia/datasets/Horse10_AL_infl_OH'
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

gpu_to_use = 1 # with gpu faster but I get out of memory error
alexnet_node_output_str = 'classifier.2' # alexnet layer to get feature map at (for us, output of fc6)
dataloader_params = {'batch_size': 64,
                     'shuffle': False,
                     'num_workers': 6}

output_ranked_idcs_pickle_path = os.path.join(os.path.dirname(path_to_pickle_w_base_idcs),
                                              'horses_AL_OH_train_infl_ranked_idcs.pkl')

###############################################################################
# %% Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [map_shuffle_id_to_base_train_idcs,
      map_shuffle_id_to_AL_train_idcs,
      map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)

############################################################
# %% Create dataloader for full dataset
alexnet_preprocess = transforms.Compose([transforms.Resize(224), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225]),])
dataset = CustomImageDataset(pd.read_hdf(path_to_h5_file), #path_to_h5_file,
                             os.path.join(reference_dir_path),
                             transform=alexnet_preprocess)

params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}
dataloader = torch.utils.data.DataLoader(dataset, **params)

################################################
# %% Select GPU or CPU for alexnet inference
use_cuda = torch.cuda.is_available()
if use_cuda and gpu_to_use != None:
    device = torch.device("cuda:{}".format(gpu_to_use))
else:
    device = 'cpu'
torch.backends.cudnn.benchmark = True
print(device)
#########################################################
# %% Get AlexNet feature extractor

# fetch model
alexnet_model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
alexnet_feature_extractor = create_feature_extractor(alexnet_model, 
                                                    return_nodes=[alexnet_node_output_str]) # see print(alexnet_model)
# do I need eval()? 
alexnet_feature_extractor.eval()
alexnet_feature_extractor.to(device)

##############################################################
# %% Extract feature vectors for the whole dataset

# Forward pass
list_feature_tensors_per_img = []
for data_batch in tqdm(dataloader):
    with torch.no_grad():
        out = alexnet_feature_extractor(data_batch.to(device))
    list_feature_tensors_per_img.append(out[alexnet_node_output_str]) 
feature_tensors = torch.cat(list_feature_tensors_per_img)
feature_arrays = feature_tensors.cpu().numpy() # (8114, 4096)

#########################################################
# %% For every shuffle, rank AL train idcs by 
# - influence, aka sum-of-distances to other AL train samples (ref: rest of AL train idcs)
# - next: diversity, aka sum-of-distances to already seen train samples (OJO changes with iterations)

map_shuffle_id_to_AL_train_idcs_ranked = dict()
flag_sort_from_max_to_min = False # if False, sorted in ascending order

NUM_SHUFFLES = len(map_shuffle_id_to_AL_train_idcs.keys())
for sh in range(1,NUM_SHUFFLES+1):
    
    # #------------------------------------------------------
    # Define query idcs and reference idcd
    query_idcs = map_shuffle_id_to_AL_train_idcs[sh]
    ref_idcs = map_shuffle_id_to_AL_train_idcs[sh]

    # if ref_idcs were already seen idcs: diff for each AL model.... 
    #------------------------------------------------------
    # - Compute cosine distance (1-cos_similarity)) between feature vectors
    # feature_arrays = feature_tensors.numpy() # feature_tensors.cpu().numpy() # nrows= observations, ncols=dimensions of space

    dist_array_1 = pairwise_cosine_distance_two_inputs(feature_arrays[query_idcs,:],
                                                       feature_arrays[ref_idcs,:])

    # # OJO! cosine distance between vectors u and v computed as 1 - cos(angle between u,v)
    dist_array = cdist(feature_arrays[query_idcs,:], # rows in output matrix- query
                       feature_arrays[ref_idcs,:], # cols - ref?
                       'cosine')

    # check same result
    # ------------------------------------------------------------------                   
    # - Rank idcs by their sum of cosine distance to others
    sum_of_dist_per_query_idx  = np.sum(dist_array, axis=1)
    list_query_idcs_ranked = [id for id, sum_dists in sorted(zip(query_idcs,
                                                                 sum_of_dist_per_query_idx),
                                                        key=lambda pair: pair[1],
                                                        reverse=flag_sort_from_max_to_min)] 

    # save
    map_shuffle_id_to_AL_train_idcs_ranked[sh] = list_query_idcs_ranked

# %% save for timeit

with open('feature_tensors.pkl','wb') as file:
     pickle.dump(feature_tensors,file) #feature_tensors
with open('feature_arrays.pkl','wb') as file:
     pickle.dump(feature_arrays,file) #feature_tensors


#####################################################################
# %% Save data
# idcs per shuffle
with open(output_ranked_idcs_pickle_path,'wb') as file:
    pickle.dump(map_shuffle_id_to_AL_train_idcs_ranked, file)

#####################################################################
# %%
# to check which frames are in the top 5
# df_groundtruth = pd.read_hdf(path_to_h5_file)

# df_groundtruth.index[map_shuffle_id_to_AL_train_idcs_ranked[1][:5]]