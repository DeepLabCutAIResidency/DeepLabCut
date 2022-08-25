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

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import set_inference_params_in_test_cfg 
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import setup_TF_graph_for_inference
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_batch_scmaps_per_frame
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_mpe_per_bdprt_and_frame
from deeplabcut.active_learning_iframes.mpe_horse_dataset_utils import compute_mpe_per_frame
#########################################################################################
# %%
# Inputs

# parent directory data
reference_dir_path = '/home/sofia/datasets/Horse10_AL_uncert_OH' 
path_to_pickle_w_base_idcs = os.path.join(reference_dir_path,
                                          'horses_AL_OH_train_test_idcs_split.pkl') #TODO these should probably be a unique file, not copies over each AL approach
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

# models subdirectory prefix
model_subdir_prefix = 'Horse10_AL_uncert{0:0=3d}' # subdirs with suffix _AL_unif{}, where {}=n frames from active learning
list_fraction_AL_frames = [25, 50, 75, 100] # [0,10,50,100,500] # number of frames to sample from AL test set and pass to train set

# evaluation of model's uncertainty on images- use Horse10_AL_unif000 models to run inference
path_to_model_for_uncert_evaluation = '/home/sofia/datasets/Horse10_AL_unif_OH/Horse10_AL_unif000' # 'Horse10_AL_unif000_TEST' #----------
cfg_path_for_uncert_snapshot = os.path.join(path_to_model_for_uncert_evaluation,
                                            'config.yaml') # common to all shuffles
gpu_to_use = 0
snapshot_idx = 0 # typically snapshots are saved at the following training iters: 50k, 10k, 150k, 200k

# uncertainty metric params (MPE)
batch_size_inference = 4 # cfg["batch_size"]; to compute scoremaps
downsampled_img_ny_nx_nc = (162, 288, 3) # common desired size to all images (ChesnutHorseLight is double res than the rest!)
min_px_btw_peaks = 2 # Peaks are the local maxima in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`).
min_peak_intensity = 0 #.001 #0.001 # 0.001
max_n_peaks = 5 # float('inf')
mpe_metric_per_frame_str = 'max' # choose from ['mean','max','median']

# train config template (with adam params)
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


###########################################################
# %%
# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [map_shuffle_id_to_base_train_idcs,
      map_shuffle_id_to_AL_train_idcs,
      map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)

###########################################################
# %%
## Create  dir structure for each model

# list of models = list of fractions of active learning frames to add to train set 
for fr_AL_samples in list_fraction_AL_frames:
    model_dir_path = os.path.join(reference_dir_path, model_subdir_prefix.format(fr_AL_samples))
    # copy labeled-data tree
    shutil.copytree(os.path.join(reference_dir_path,'labeled-data'),
                    os.path.join(model_dir_path,'labeled-data'))  
    # copy config
    shutil.copyfile(os.path.join(reference_dir_path,'config.yaml'),
                    os.path.join(model_dir_path,'config.yaml'))


###########################################################
# %% Compute uncertainty of AL train samples,
# using the model trained on the base train samples + 0% of AL train samples
# TODO: this may be better as a separate script

df_groundtruth = pd.read_hdf(path_to_h5_file)
map_shuffle_id_to_AL_train_idcs_ranked = dict()
NUM_SHUFFLES = len(map_shuffle_id_to_base_train_idcs.keys())
for sh in range(1,NUM_SHUFFLES+1):
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

    #---------------------------------
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
    #---------------------------------   
    # Sort idcs by MPE metric and save 
    list_AL_train_idcs_ranked =[id for id, mean_mpe in sorted(zip(map_shuffle_id_to_AL_train_idcs[sh], #idcs from AL train set
                                                                  mpe_metrics_per_frame[mpe_metric_per_frame_str]),
                                                        key=lambda pair: pair[1],
                                                        reverse=True)] # sort by the second element of the tuple in desc order
    # list_AL_train_images_sorted_by_mean_mpe = list(df_groundtruth.index[list_idcs_ranked])
    map_shuffle_id_to_AL_train_idcs_ranked[sh] = list_AL_train_idcs_ranked #idx_AL_train_idcs_to_transfer
#########################################################################################
# %% Save pickle... for plot mostly
# path_to_pickle_w_AL_train_idcs_ranked_by_infl = os.path.join(reference_dir_path,
#                                                              'horses_AL_OH_train_uncert_ranked_idcs.pkl')

# with open(path_to_pickle_w_AL_train_idcs_ranked_by_infl,'wb') as file:
#     pickle.dump(map_shuffle_id_to_AL_train_idcs_ranked, file)   
#                                                          
#########################################################################################
# %%
# Create training datasets
# ATT! idcs refer to dataframe h5 file in training-datasets dir, as it is before runnin create_training_dataset!!! 
# So I need to replace it
# NUM_SHUFFLES = len(map_shuffle_id_to_base_train_idcs.keys())

# list of models = list of number of active learning frames to add to train set 
for fr_AL_samples in list_fraction_AL_frames:
    print('------------------------------------------')
    print('Model with {}% of active learning frames sampled'.format(fr_AL_samples))

    ## Get model dir path and model config
    model_dir_path = os.path.join(reference_dir_path, 
                                  model_subdir_prefix.format(fr_AL_samples)) 
    config_path = os.path.join(model_dir_path,'config.yaml') 

    ###########################################################
    ## Create list of train and test idcs per shuffle for this model (list of lists)
    train_idcs_one_model = []
    test_idcs_one_model = []
    list_training_fraction_per_shuffle = []
    for sh in range(1,NUM_SHUFFLES+1):
        # get list of base train for this shuffle
        list_base_train_idcs = map_shuffle_id_to_base_train_idcs[sh] 
        
        #---------------------------------------------------------------------------
        # get list of top x% AL train idcs, ranked by desired mpe metric
        n_AL_samples = math.floor(fr_AL_samples*len(map_shuffle_id_to_AL_train_idcs_ranked[sh])/100)
        list_AL_train_idcs_to_transfer = map_shuffle_id_to_AL_train_idcs_ranked[sh][:n_AL_samples]
        #---------------------------------------------------------------------------

        ## Compute final lists of train indices for this shuffle, after transfer
        list_final_train_idcs = list_base_train_idcs + \
                                list_AL_train_idcs_to_transfer
        # append results to lists of lists
        train_idcs_one_model.append(list_final_train_idcs)

        ## Compute test indices
        list_final_test_idcs = map_shuffle_id_to_OOD_test_idcs[sh]
        test_idcs_one_model.append(list_final_test_idcs)

        ## Compute training fraction for each shuffle
        print('Shuffle {} train idcs: {}'.format(sh,len(list_final_train_idcs)))
        print('Shuffle {} test idcs: {}'.format(sh,len(list_final_test_idcs)))
        training_fraction_one_shuffle = \
            round(len(list_final_train_idcs) * 1.0 / (len(list_final_train_idcs)+len(list_final_test_idcs)), 2)
        training_fraction_one_shuffle = int(100*training_fraction_one_shuffle)/100 # to match name of shuffle saved!
                                                # int((len(list_final_train_idcs)/\
                                                #   (len(list_final_train_idcs)+len(list_final_test_idcs)))*100)/100
        list_training_fraction_per_shuffle.append(training_fraction_one_shuffle)

    ############################################################
    ## Edit training fraction in this model's config file
    training_fraction_dict = {'TrainingFraction': list_training_fraction_per_shuffle}
    edit_config(config_path,
                training_fraction_dict)

    ###########################################################
    ## Create training dataset for this model (all shuffles)
    create_training_dataset(config_path,
                            num_shuffles=NUM_SHUFFLES,
                            trainIndices=train_idcs_one_model,  #map_shuffle_id_to_train_idcs[sh],
                            testIndices=test_idcs_one_model, # passing IID test indices for now
                            posecfg_template=pose_cfg_yaml_adam_path) # augmenter_type=None, posecfg_template=None,

    ###########################################################
    ## Copy h5 file from 'training-datasets' at reference dir, to newly created 'training-datasets' directory for this model 
    # it is overwritten after create_training_dataset() so I need to get it back from reference parent dir !   
    for ext in ['.csv','.h5']:                         
        shutil.copyfile(os.path.join(reference_dir_path,'training-datasets',
                                    'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron{}'.format(ext)), # src
                        os.path.join(model_dir_path,'training-datasets',
                                    'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron{}'.format(ext))) # dest



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
