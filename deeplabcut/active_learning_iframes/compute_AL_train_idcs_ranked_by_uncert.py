'''

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

## Base idcs data and groundtruth
#TODO these should probably be a unique file, not copies over each AL approach
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_AL_unif_OH/horses_AL_OH_train_test_idcs_split.pkl'
reference_dir_path = '/home/sofia/datasets/Horse10_AL_unif_OH'
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

## output path for pickle with uncert ranked idcs
path_to_output_pickle_w_ranked_idcs = os.path.join(os.path.dirname(path_to_pickle_w_base_idcs),
                                                   'horses_AL_OH_train_uncert_ranked_idcs.pkl')

## model to use to evaluate uncertainty
# (use Horse10_AL_unif000 models, aka models trained on 1 horse only, to run inference)
path_to_model_for_uncert_evaluation = '/home/sofia/datasets/Horse10_AL_unif_OH/Horse10_AL_unif000' 
cfg_path_for_uncert_snapshot = os.path.join(path_to_model_for_uncert_evaluation,
                                            'config.yaml') # common to all shuffles
gpu_to_use = 0 
snapshot_idx = 0 # typically snapshots are saved at the following training iters: 50k, 10k, 150k, 200k

# MPE computations params 
batch_size_inference = 4 # cfg["batch_size"]; to compute scoremaps
downsampled_img_ny_nx_nc = (162, 288, 3) # common desired size to all images (ChesnutHorseLight is double res than the rest!)
min_px_btw_peaks = 2 # Peaks are the local maxima in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`).
min_peak_intensity = 0 #.001 #0.001 # 0.001
max_n_peaks = 5 # float('inf')
mpe_metric_per_frame_str = 'max' # choose from ['mean','max','median']


# train config template (with adam params)
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


###########################################################
# %% Compute uncertainty of AL train samples, and sample
# using the model trained on the base train samples + 0% of AL train samples
# TODO: this may be better as a separate script

# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [_,map_shuffle_id_to_AL_train_idcs, _] = pickle.load(file)

# Get groundtruth
df_groundtruth = pd.read_hdf(path_to_h5_file)     

# Compute ranked idcs per shuffle
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
    # Sort idcs by MPE metric and save 
    list_AL_train_idcs_ranked =[id for id, mean_mpe in sorted(zip(map_shuffle_id_to_AL_train_idcs[sh], #idcs from AL train set
                                                                  mpe_metrics_per_frame[mpe_metric_per_frame_str]),
                                                        key=lambda pair: pair[1],
                                                        reverse=True)] # sort by the second element of the tuple in desc order
    # list_AL_train_images_sorted_by_mean_mpe = list(df_groundtruth.index[list_idcs_ranked])
    map_shuffle_id_to_AL_train_idcs_ranked[sh] = list_AL_train_idcs_ranked

#####################################################################
# %% Save data
# idcs per shuffle
with open(path_to_output_pickle_w_ranked_idcs,'wb') as file:
    pickle.dump(map_shuffle_id_to_AL_train_idcs_ranked, file)



# # %%
# #########################################
# # Check AL000: picke idcs against df data
# fr_AL_samples = 25
# shuffle_id = 1

# model_dir_path = os.path.join(reference_dir_path, 
#                               model_subdir_prefix.format(fr_AL_samples)) 
# df = pd.read_hdf(os.path.join(model_dir_path,
#                              'training-datasets/iteration-0/'+\
#                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'))

# config_path = os.path.join(model_dir_path,'config.yaml') 
# cfg = read_config(config_path)
# trainFraction = cfg['TrainingFraction'][shuffle_id-1] # ATT! shuffle_id starts with 1!
# path_to_shuffle_pickle = os.path.join(model_dir_path,
#                                       'training-datasets/iteration-0/'+\
#                                       'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_{}shuffle{}.pickle'.format(int(round(trainFraction*100,6)),
#                                                                                                                           shuffle_id)) #53-1, 54-2, 50-3

# with open(path_to_shuffle_pickle, "rb") as f:
#     pickledata = pickle.load(f)
#     data = pickledata[0] 
#     train_idcs = pickledata[1] 
#     test_idcs = pickledata[2] 
#     split_fraction = pickledata[3] # 

# print('-----')
# print('Model with {}% of AL frames, shuffle {}'.format(fr_AL_samples,shuffle_id))
# print('-----')
# print('Number of training samples = {}'.format(len(train_idcs))) 
# print('Number of test samples = {}'.format(len(test_idcs)))
# print('-----')
# list_horses_in_train = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in train_idcs]))
# list_horses_in_train.sort()
# print('Number of horses in train set (incl AL) = {}'.format(len(list_horses_in_train))) 
# print(list_horses_in_train)
# list_horses_in_test = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in test_idcs]))
# list_horses_in_test.sort()
# print('Number of horses in test set = {}'.format(len(list_horses_in_test))) # ok
# print(list_horses_in_test)


# # %%
