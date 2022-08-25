"""
Compute MPE per bodypart for the OOD test images, and save results for each shuffle
"""

#########################################
# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import re
import math

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

import os, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import argparse
import yaml
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # to supress future warnings...

from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import set_inference_params_in_test_cfg 
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import setup_TF_graph_for_inference
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_batch_scmaps_per_frame
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_mpe_per_bdprt_and_frame
#########################################
# %%
# Input params
# Base train and test idcs
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_AL_unif_OH/horses_AL_OH_train_test_idcs_split.pkl'

# Groundtruth
path_to_h5_file = os.path.join(os.path.dirname(path_to_pickle_w_base_idcs),  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!


# model to use for inference to compute uncertainty on images- use Horse10_AL_unif000 models to run inference
path_to_model_for_uncert_evaluation = '/home/sofia/datasets/Horse10_AL_unif_OH/Horse10_AL_unif000' 
cfg_path_for_uncert_snapshot = os.path.join(path_to_model_for_uncert_evaluation,
                                            'config.yaml') # common to all shuffles
gpu_to_use = 1
snapshot_idx = 0 # typically snapshots are saved at the following training iters: 50k, 10k, 150k, 200k

# uncertainty metric params (MPE)
batch_size_inference = 4 # cfg["batch_size"]; to compute scoremaps
downsampled_img_ny_nx_nc = (162, 288, 3) # common desired size to all images (ChesnutHorseLight is double res than the rest!)
min_px_btw_peaks = 2 # Peaks are the local maxima in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`).
min_peak_intensity = 0 #.001 #0.001 # 0.001
max_n_peaks = 5 # float('inf')
mpe_metric_per_frame_str = 'max' # choose from ['mean','max','median']

# output
snapshot_idx_str = str(snapshot_idx)
if snapshot_idx_str.startswith('-'):
    snapshot_idx_str = 'neg' + snapshot_idx_str[1:]
pickle_output_path = os.path.join(os.path.dirname(path_to_pickle_w_base_idcs),
                                  f'horses_AL_OH_mpe_per_shuffle_snapshotID_{snapshot_idx_str}.pkl')

TRAIN_ITERATION = 0 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?
NUM_SHUFFLES = 3

###########################################################
# %%
# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [_, #map_shuffle_id_to_base_train_idcs,
      _, #map_shuffle_id_to_AL_train_idcs,
      map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)

# Get groundtruth data
df_groundtruth = pd.read_hdf(path_to_h5_file)
# list_bdprts = list(set([x[1] for x in df_groundtruth.columns]))

# list_bdprts_rep_from_df = [x[1] for x in df_groundtruth.columns] # from original df columns, with repetitions
# list_bdprt_loc_in_orig_df = [list_bdprts_rep_from_df.index(y) for y in list_bdprts]
# list_bdprts_srted_as_df_cols = [x for x,_ in sorted(zip(list_bdprts,list_bdprt_loc_in_orig_df),
#                                                     key=lambda pair:pair[1])]

## Set 'allow growth' before eval (allow growth bug)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#############################################################################
# %% Add MPE per bodypart for the OOD test images
dict_df_test_only_w_uncert_per_shuffle = dict()

for sh in range(1,NUM_SHUFFLES+1):    
    # Set inference params in test config: init_weights (= snapshot), batchsize and number of outputs
    # test config is diff per shuffle
    dlc_cfg, cfg = set_inference_params_in_test_cfg(cfg_path_for_uncert_snapshot,
                                                    batch_size_inference,
                                                    sh,
                                                    trainingsetindex = None, # if None, extracted from config based on shuffle number
                                                    modelprefix='',
                                                    snapshotindex=snapshot_idx)

    # Setup TF graph
    sess, inputs, outputs = setup_TF_graph_for_inference(dlc_cfg, #dlc_cfg: # pass config loaded, not path (use load_config())
                                                         gpu_to_use)

    # Prepare batch of images to run inference on (OJO, different per shuffle)
    list_OOD_test_images = list(df_groundtruth.index[map_shuffle_id_to_OOD_test_idcs[sh]])

    ### Run inference on AL train images
    scmaps_all_OOD_frames = compute_batch_scmaps_per_frame(cfg, 
                                                           dlc_cfg, 
                                                           sess, inputs, outputs, 
                                                           os.path.dirname(cfg_path_for_uncert_snapshot), 
                                                           list_OOD_test_images, 
                                                           downsampled_img_ny_nx_nc,
                                                           batch_size_inference) 
    # scmaps_all_OOD_frames.shape = (5185, 22, 36, 22) = (nsamples in OOD set, scoremap_rows, scoremap_cols, n_bdprts)

    ### Compute uncertainty (MPE) per bodypart and mean/max per frame
    mpe_per_frame_and_bprt, _, _, _ = compute_mpe_per_bdprt_and_frame(scmaps_all_OOD_frames,
                                                                        min_px_btw_peaks,
                                                                        min_peak_intensity,
                                                                        max_n_peaks)
    # mpe_per_frame_and_bprt.shape = (5185, 22)

    # compute mean, max and median over all bodyparts per frame                                                             
    mpe_metrics_per_frame = {'mean': np.mean(mpe_per_frame_and_bprt,axis=-1),
                             'max': np.max(mpe_per_frame_and_bprt,axis=-1),
                             'median': np.median(mpe_per_frame_and_bprt,axis=-1)}   

    # add results to dataframe of OOD test data only
    df_groundtruth_test_only = df_groundtruth.iloc[map_shuffle_id_to_OOD_test_idcs[sh],:] 
#     # check list of images matches
#     if df_groundtruth_test_only.index.to_list() != list_OOD_test_images:
#         print('ERROR: dataframes with human labels and model predictions have different rows')
#         sys.exit()
    # add mpe per bodypart
    for bp_i, bp_str in enumerate(dlc_cfg['all_joints_names']):
        df_groundtruth_test_only.loc[:,(cfg['scorer'],bp_str,'MPE_bdprt')] = mpe_per_frame_and_bprt[:,bp_i]
    # add mpe metrics per frame
    for ky in mpe_metrics_per_frame.keys():
        df_groundtruth_test_only.loc[:,f'{ky}_MPE'] = mpe_metrics_per_frame[ky]

    # save to dict 
    dict_df_test_only_w_uncert_per_shuffle[sh]  =  df_groundtruth_test_only                                                                                

##############################################

# %%
# #############################################
# # %%
with open(pickle_output_path,'wb') as file:
    pickle.dump([dict_df_test_only_w_uncert_per_shuffle], file)

# ##########################################


# %%
