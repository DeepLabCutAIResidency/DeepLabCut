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
from sklearn.cluster import KMeans
import itertools 


from deeplabcut.active_learning_iframes.infl_horse_dataset_utils import CustomImageDataset, pairwise_cosine_distance, pairwise_cosine_distance_two_inputs
#####################################################
# %%

## Base idcs data and groundtruth
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_test_idcs_split.pkl' 
reference_dir_path = '/home/sofia/datasets/Horse10_AL_unif_OH'
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

## output path for pickle with uncert+kmeans ranked idcs
path_to_output_pickle_w_ranked_idcs = os.path.join(os.path.dirname(path_to_pickle_w_base_idcs),
                                                   'horses_AL_OH_train_infl_kmeans_ranked_idcs.pkl')

## model used to evaluate influence
gpu_to_use = 2 # with gpu faster but I get out of memory error
alexnet_node_output_str = 'classifier.2' # alexnet layer to get feature map at (for us, output of fc6)
alexnet_img_transform_mean = [0.485, 0.456, 0.406] # see https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet_img_transform_std = [0.229, 0.224, 0.225] # see https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet_dataloader_params = {'batch_size': 64,
                            'shuffle': False,
                            'num_workers': 6}

## k-means clustering on alexnet features
kmeans_n_clusters = 9 # expected number of horses
kmeans_max_iter = 1000
kmeans_random_state = 0



############################################################
# %% Compute AlexNet features for the full h5 dataset
# TODO: make this a fn, common to all approaches using infl or kmeans approach
#-----------------------------------
## Create dataloader for full dataset
df_groundtruth = pd.read_hdf(path_to_h5_file)  

alexnet_preprocess = transforms.Compose([transforms.Resize(224), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=alexnet_img_transform_mean, 
                                                              std=alexnet_img_transform_std),])                                                            
dataset = CustomImageDataset(df_groundtruth, #path_to_h5_file,
                             os.path.join(reference_dir_path),
                             transform=alexnet_preprocess)

dataloader = torch.utils.data.DataLoader(dataset, **alexnet_dataloader_params)

#-----------------------------------
## Select GPU or CPU for alexnet inference
use_cuda = torch.cuda.is_available()
if use_cuda and gpu_to_use != None:
    device = torch.device("cuda:{}".format(gpu_to_use))
else:
    device = 'cpu'
torch.backends.cudnn.benchmark = True
print(device)

#-----------------------------------
## Get AlexNet as a feature extractor
# fetch model
alexnet_model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
alexnet_feature_extractor = create_feature_extractor(alexnet_model, 
                                                    return_nodes=[alexnet_node_output_str]) # see print(alexnet_model)
# do I need eval()? 
alexnet_feature_extractor.eval() # I think so, so that it doesn't do dropout?
alexnet_feature_extractor.to(device)

#-----------------------------------
## Extract feature vectors for the whole dataset
list_feature_tensors_per_img = []
for data_batch in tqdm(dataloader):
    with torch.no_grad():
        out = alexnet_feature_extractor(data_batch.to(device))
    list_feature_tensors_per_img.append(out[alexnet_node_output_str]) 
alexnet_feature_tensors = torch.cat(list_feature_tensors_per_img)
alexnet_feature_arrays = alexnet_feature_tensors.cpu().numpy() # (8114, 4096)---row indexing should match df from h5 file


#########################################################
# %% For every shuffle, rank AL train idcs 
# - by influence, aka sum-of-distances to other AL train samples (ref: rest of AL train idcs)
# - sample across clusters to avoid 'repetition

# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [_,map_shuffle_id_to_AL_train_idcs,_] = pickle.load(file)

# Compute ranked idcs per shuffle
map_shuffle_id_to_AL_train_idcs_ranked = dict()
flag_sort_from_max_to_min = False # if False, sorted in ascending order
NUM_SHUFFLES = len(map_shuffle_id_to_AL_train_idcs.keys())
for sh in range(1,NUM_SHUFFLES+1):

    ##################################################
    ## Extract feature vectors for AL train images and run kmeans                                                                            
    alexnet_feature_vectors_AL_train_imgs = alexnet_feature_arrays[map_shuffle_id_to_AL_train_idcs[sh],:]
    # run k-means
    kmeans_results = KMeans(n_clusters=kmeans_n_clusters, 
                            max_iter=kmeans_max_iter,
                            random_state=kmeans_random_state).fit(alexnet_feature_vectors_AL_train_imgs)
    kmeans_id = kmeans_results.labels_

    #########################################
    # - Compute cosine distance (1-cos_similarity)) between feature vectors
    # TODO: use pytorch functions instead of  numpy for faster computation
    # feature_arrays = feature_tensors.numpy() # feature_tensors.cpu().numpy() # nrows= observations, ncols=dimensions of space
    
    ## Define query idcs and reference idcs
    # here they are the same, eventually we may want to use different query and reference indices
    # for ex: reference indices may be all the data already seen by the model in training 
    query_idcs = map_shuffle_id_to_AL_train_idcs[sh]
    ref_idcs = map_shuffle_id_to_AL_train_idcs[sh]

    # Pytorch approach: TODO check I get same result as numpy
    # dist_tensor = pairwise_cosine_distance(alexnet_feature_tensors[query_idcs,:])
    # dist_array = dist_tensor.cpu().numpy()
    #
    # dist_array_1 = pairwise_cosine_distance_two_inputs(alexnet_feature_tensors[query_idcs,:],
    #                                                    alexnet_feature_tensors[ref_idcs,:])

    # # OJO! cosine distance between vectors u and v computed as [1 - cos(angle between u,v)]
    dist_array = cdist(alexnet_feature_arrays[query_idcs,:], # rows in output matrix- query
                       alexnet_feature_arrays[ref_idcs,:], # cols - ref?
                       'cosine')

    # # check same result between numpy and pytorch fn
    # Option 1:
    # dist_array_1 = pairwise_cosine_distance_two_inputs(alexnet_feature_tensors[query_idcs,:],
    #                                                    alexnet_feature_tensors[ref_idcs,:])
    # dist_array_1_cpu=dist_array_1.cpu().numpy()
    # np.max(np.abs(dist_array_1_cpu - dist_array))---->0.422 : not the same! review
    #
    # Option 2:
    # dist_array_1 = pairwise_cosine_distance(alexnet_feature_tensors[query_idcs,:])
    #
    # dist_array_1_cpu=dist_array_1.cpu().numpy()
    # np.max(np.abs(dist_array_1_cpu - dist_array))---->1.2307362682317802e-06, ok?
    #########################################            
    # - Compute sum of cosine distance to other samples
    sum_of_dist_per_query_idx  = np.sum(dist_array, axis=1)
    # list_query_idcs_ranked = [id for id, sum_dists in sorted(zip(query_idcs,
    #                                                              sum_of_dist_per_query_idx),
    #                                                     key=lambda pair: pair[1],
    #                                                     reverse=flag_sort_from_max_to_min)] 

    ##########################################################
    ## Combine ranking metric with index in h5 dataframe and with kmeans label
    # Build dataframe with h5_df_idx, k_means_idx,  and MPE for each image
    df_sampling_AL_train_one_shuffle = pd.DataFrame({'idx_h5':np.array(map_shuffle_id_to_AL_train_idcs[sh]),
                                                     'kmeans_label':np.array(kmeans_id),
                                                     'sum_distances':sum_of_dist_per_query_idx})
    # sort by 'sum of distances' to other samples in ascending order
    df_sampling_AL_train_one_shuffle.sort_values(by='sum_distances',
                                                 axis='rows',
                                                 ascending=True, # OJO from low to high distance!
                                                 inplace=True)             
    df_sampling_AL_train_one_shuffle.reset_index(inplace=True,
                                                drop=True)

    ##########################################
    ## Sample ranked frames across kmeans clusters
    map_kmeans_label_to_df_sampling = dict()
    for k in set(kmeans_id):
        map_kmeans_label_to_df_sampling[k] = \
            df_sampling_AL_train_one_shuffle[df_sampling_AL_train_one_shuffle['kmeans_label'] == k]

    ## Prepare idcs for sampling
    # get idcs in h5 dataframe for each kmeans cluster, sorted by 'sum of distances'
    list_of_ranked_AL_train_idcs_per_kmeans_label = \
        [list(x.loc[:,'idx_h5']) for x in map_kmeans_label_to_df_sampling.values()]
    # append nans to match lists' lengths
    max_len = max([len(x) for x in list_of_ranked_AL_train_idcs_per_kmeans_label])
    list_of_ranked_AL_train_idcs_per_kmeans_label_filled = \
        [x + [np.nan]*abs(len(x)-max_len) for x in list_of_ranked_AL_train_idcs_per_kmeans_label] 
    # interleave                                                        
    list_AL_train_idcs_ranked_interleaved = list(itertools.chain(*zip(*list_of_ranked_AL_train_idcs_per_kmeans_label_filled)))
    # remove nans
    list_AL_train_idcs_ranked_interleaved = [x for x in list_AL_train_idcs_ranked_interleaved if not np.isnan(float(x))]
    
    ########################################
    # save
    map_shuffle_id_to_AL_train_idcs_ranked[sh] = list_AL_train_idcs_ranked_interleaved

#####################################################################
# %% save for timeit

# with open('feature_tensors.pkl','wb') as file:
#      pickle.dump(alexnet_feature_tensors,file) #feature_tensors
# with open('feature_arrays.pkl','wb') as file:
#      pickle.dump(alexnet_feature_arrays,file) #feature_tensors


#####################################################################
# %% Save data
# idcs per shuffle
with open(path_to_output_pickle_w_ranked_idcs,'wb') as file:
    pickle.dump(map_shuffle_id_to_AL_train_idcs_ranked, file)

#####################################################################
# %%
# to check which frames are in the top 5
# df_groundtruth = pd.read_hdf(path_to_h5_file)

# df_groundtruth.index[map_shuffle_id_to_AL_train_idcs_ranked[1][:5]]