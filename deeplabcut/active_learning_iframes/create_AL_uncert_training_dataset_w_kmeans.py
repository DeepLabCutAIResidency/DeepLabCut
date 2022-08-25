'''
This script prepares the training datasets for the active learning baseline, using the OneHorse approach. 
- In this approach, we separate the data into 10 train horses and 20 test horses (one horse = one video)
- We divide the 10 train horses in two parts: a base train set (one horse) and an active train set 
    (the remaining 9 horses from the full train set)
- We train N models with N different active learning setups
    We train on the base train set +  x% of frames from the AL train set
    We typicall consider x = [0, 25, 50, 75, 100]
- How are the AL frames selected?
    In the baseline, the AL train frames are selected via uniform temporal sampling across all videos
    Here, the AL train frames are ranked based on the model's uncertainty. 
    The model used to evaluate the AL train frames uncertainty is the one trained only on the base indices
    
    
- The test idcs passed per shuffle are the frames in the testOOD set

- The sets of train indices (for both base and AL sets) and test indices per shuffle
  are loaded from the pickle file 'path_to_pickle_w_base_idcs'

- Before running this script:
    - Download horse10.tar.gz and extract in 'Horse10_AL_unif': (use --strip-components 1)
        mkdir Horse10_AL_unif
        tar -xvzf horse10.tar.gz -C /home/sofia/datasets/Horse10_AL_unif --strip-components 1


'''


# %%
import os, sys, shutil
import re 
import argparse
import yaml
import deeplabcut
import pickle

import pandas as pd
import numpy as np
import math
# import random
from tqdm import tqdm
from sklearn.cluster import KMeans
import itertools 

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor # get_graph_node_names

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset
from deeplabcut.active_learning_iframes.infl_horse_dataset_utils import CustomImageDataset


from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import set_inference_params_in_test_cfg 
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import setup_TF_graph_for_inference
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_batch_scmaps_per_frame
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_mpe_per_bdprt_and_frame
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_mpe_per_frame
#########################################################################################
# %%
# Inputs

# # parent directory data---- these should be agnostic to AL strategy
reference_dir_path = '/home/sofia/datasets/Horse10_AL_uncert_OH' 
path_to_pickle_w_base_idcs = os.path.join(reference_dir_path,
                                          'horses_AL_OH_train_test_idcs_split.pkl') #TODO these should probably be a unique file, not copies over each AL approach
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

# output path for pickle
path_to_output_pickle_w_ranked_idcs = ''

# evaluation of images uncertainty on images- use Horse10_AL_unif000 models (aka models trained on 1 horse) to run inference
path_to_model_for_uncert_evaluation = '/home/sofia/datasets/Horse10_AL_unif_OH/Horse10_AL_unif000' 
cfg_path_for_uncert_snapshot = os.path.join(path_to_model_for_uncert_evaluation,
                                            'config.yaml') # common to all shuffles
gpu_to_use = 0 # also for alexnet
snapshot_idx = 0 # typically snapshots are saved at the following training iters: 50k, 10k, 150k, 200k

# MPE computations params 
batch_size_inference = 4 # cfg["batch_size"]; to compute scoremaps
downsampled_img_ny_nx_nc = (162, 288, 3) # common desired size to all images (ChesnutHorseLight is double res than the rest!)
min_px_btw_peaks = 2 # Peaks are the local maxima in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`).
min_peak_intensity = 0 #.001 #0.001 # 0.001
max_n_peaks = 5 # float('inf')
mpe_metric_per_frame_str = 'max' # choose from ['mean','max','median']


# images as alexnet feature vectors
alexnet_node_output_str = 'classifier.2' # alexnet layer to get feature map at (for us, output of fc6)
alexnet_img_transform_mean = [0.485, 0.456, 0.406] # see https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet_img_transform_std = [0.229, 0.224, 0.225] # see https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet_dataloader_params = {'batch_size': 64,
                            'shuffle': False,
                            'num_workers': 6}

# kmeans on alexnet features
kmeans_n_clusters = 9 # expected number of horses
kmeans_max_iter = 1000
kmeans_random_state = 0

# train config template (with adam params)
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


############################################################
# %% Compute AlexNet features for the full h5 datasett
#-----------------------------------
## Create dataloader for full dataset
alexnet_preprocess = transforms.Compose([transforms.Resize(224), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=alexnet_img_transform_mean, 
                                                              std=alexnet_img_transform_std),])
df_groundtruth = pd.read_hdf(path_to_h5_file)                                                              
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
## Get AlexNet feature extractor
# fetch model
alexnet_model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
alexnet_feature_extractor = create_feature_extractor(alexnet_model, 
                                                    return_nodes=[alexnet_node_output_str]) # see print(alexnet_model)
# do I need eval()? 
alexnet_feature_extractor.eval() # I think so, so that it doesn't do dropout
alexnet_feature_extractor.to(device)

#-----------------------------------
## Extract feature vectors for the whole dataset
list_feature_tensors_per_img = []
for data_batch in tqdm(dataloader):
    with torch.no_grad():
        out = alexnet_feature_extractor(data_batch.to(device))
    list_feature_tensors_per_img.append(out[alexnet_node_output_str]) 
feature_tensors = torch.cat(list_feature_tensors_per_img)
alexnet_feature_arrays = feature_tensors.cpu().numpy() # (8114, 4096)---row indexing should match df from h5 file



###########################################################
# %% Compute uncertainty of AL train samples, and sample
# using the model trained on the base train samples + 0% of AL train samples
# TODO: this may be better as a separate script

# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [_,map_shuffle_id_to_AL_train_idcs, _] = pickle.load(file)


map_shuffle_id_to_AL_train_idcs_ranked = dict()
NUM_SHUFFLES = len(map_shuffle_id_to_AL_train_idcs.keys())
for sh in range(1,NUM_SHUFFLES+1):
    ###############################################
    # Compute MPE per frame, for AL train images
    # Set inference params in test config: init_weights (= snapshot), batchsize and number of outputs
    dlc_cfg, cfg = set_inference_params_in_test_cfg(cfg_path_for_uncert_snapshot,
                                                    batch_size_inference,
                                                    sh,
                                                    trainingsetindex = None, # if None, extracted from config based on shuffle number
                                                    modelprefix='',
                                                    snapshotindex=snapshot_idx)

    # Setup TF graph
    sess, inputs, outputs = setup_TF_graph_for_inference(dlc_cfg, #dlc_cfg: # pass config loaded, not path (use load_config())
                                                         gpu_to_use)

    # Prepare batch of images for this shuffle
    list_AL_train_images = list(df_groundtruth.index[map_shuffle_id_to_AL_train_idcs[sh]])

    # Run inference on selected model and compute mean, max and median MPE per frame
    mpe_metrics_per_frame,\
    mpe_per_frame_and_bprt = compute_mpe_per_frame(cfg,
                                                    dlc_cfg,
                                                    sess, inputs, outputs,
                                                    os.path.dirname(cfg_path_for_uncert_snapshot),
                                                    list_AL_train_images,
                                                    downsampled_img_ny_nx_nc,
                                                    batch_size_inference,
                                                    min_px_btw_peaks,
                                                    min_peak_intensity,
                                                    max_n_peaks)  

    ##########################################################
    # Compute feature vectors for AL train images and run kmeans                                                                            
    alexnet_feature_vectors_AL_train_imgs = alexnet_feature_arrays[map_shuffle_id_to_AL_train_idcs[sh],:]
    # run k-means
    kmeans_results = KMeans(n_clusters=kmeans_n_clusters, 
                            max_iter=kmeans_max_iter,
                            random_state=kmeans_random_state).fit(alexnet_feature_vectors_AL_train_imgs)
    kmeans_results.cluster_centers_
    kmeans_results.labels_
    kmeans_results.inertia_ #Sum of squared distances of samples to their closest cluster center

    kmeans_id = kmeans_results.labels_

    ##########################################################
    ## Sample most uncertain frames across clusters
    #---------------------------------   
    # Build dataframe with h5_df_idx, k_means_idx,  and MPE for each image
    df_sampling_AL_train_one_shuffle = pd.DataFrame([map_shuffle_id_to_AL_train_idcs[sh],
                                                     kmeans_id,
                                                     mpe_metrics_per_frame['max']],
                                                     columns = ['idx_h5', 'kmeans_label', 'max_MPE'])
    # sort by max MPE
    df_sampling_AL_train_one_shuffle.sort_values(by='max_MPE',
                                                 axis='columns',
                                                 inplace=True)                                                 
    # split dataframes by kmeans label
    map_kmeans_label_to_df_sampling = dict()
    for k in set(kmeans_id):
        map_kmeans_label_to_df_sampling[k] = \
            df_sampling_AL_train_one_shuffle[df_sampling_AL_train_one_shuffle['kmeans_label'] == k]
    #---------------------------------   
    # Prepare idcs for sampling
    # get idcs in h5 dataframe for each kmeans cluster, sorted by MPE
    list_of_ranked_AL_train_idcs_per_kmeans_label = \
        [x.loc[:,'idx_h5'] for x in map_kmeans_label_to_df_sampling.values()]
    # append nans to match list lengths
    max_len = max([len(x) for x in list_of_ranked_AL_train_idcs_per_kmeans_label])
    list_of_ranked_AL_train_idcs_per_kmeans_label_filled = \
        [x + [np.nan]*abs(len(x)-max_len) for x in list_of_ranked_AL_train_idcs_per_kmeans_label] 
    # interleave                                                        
    list_AL_train_idcs_ranked_interleaved = list(itertools.chain(*zip(*list_of_ranked_AL_train_idcs_per_kmeans_label_filled)))
    # flatten
    # list_AL_train_idcs_ranked_interleaved = [el for lst in list_AL_train_idcs_ranked_interleaved for el in lst]
    # remove nans
    list_AL_train_idcs_ranked_interleaved = [x for x in list_AL_train_idcs_ranked_interleaved if not np.isnan(float(x))]
    
    #---------------------------------   
    # Save
    map_shuffle_id_to_AL_train_idcs_ranked[sh] = list_AL_train_idcs_ranked_interleaved #idx_AL_train_idcs_to_transfer

#####################################################################
# %% Save data
# idcs per shuffle
with open(output_ranked_idcs_pickle_path,'wb') as file:
    pickle.dump(map_shuffle_id_to_AL_train_idcs_ranked, file)






# %%
#########################################
# Check AL000: picke idcs against df data
fr_AL_samples = 25
shuffle_id = 1

model_dir_path = os.path.join(reference_dir_path, 
                              model_subdir_prefix.format(fr_AL_samples)) 
df = pd.read_hdf(os.path.join(model_dir_path,
                             'training-datasets/iteration-0/'+\
                             'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'))

config_path = os.path.join(model_dir_path,'config.yaml') 
cfg = read_config(config_path)
trainFraction = cfg['TrainingFraction'][shuffle_id-1] # ATT! shuffle_id starts with 1!
path_to_shuffle_pickle = os.path.join(model_dir_path,
                                      'training-datasets/iteration-0/'+\
                                      'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_{}shuffle{}.pickle'.format(int(round(trainFraction*100,6)),
                                                                                                                          shuffle_id)) #53-1, 54-2, 50-3

with open(path_to_shuffle_pickle, "rb") as f:
    pickledata = pickle.load(f)
    data = pickledata[0] 
    train_idcs = pickledata[1] 
    test_idcs = pickledata[2] 
    split_fraction = pickledata[3] # 

print('-----')
print('Model with {}% of AL frames, shuffle {}'.format(fr_AL_samples,shuffle_id))
print('-----')
print('Number of training samples = {}'.format(len(train_idcs))) 
print('Number of test samples = {}'.format(len(test_idcs)))
print('-----')
list_horses_in_train = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in train_idcs]))
list_horses_in_train.sort()
print('Number of horses in train set (incl AL) = {}'.format(len(list_horses_in_train))) 
print(list_horses_in_train)
list_horses_in_test = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in test_idcs]))
list_horses_in_test.sort()
print('Number of horses in test set = {}'.format(len(list_horses_in_test))) # ok
print(list_horses_in_test)


# %%
