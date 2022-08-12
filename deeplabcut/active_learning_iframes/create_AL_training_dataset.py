'''
1- Download horse10.tar.gz and extract in 'Horses10_AL2'
    2- Copy pickle file with idcs to 'Horses10_AL2' --not req
3- Rename 'Horses-Byron-2019-05-08/training-datasets' to 'Horses-Byron-2019-05-08/training-datasets_'
    4- Edit config?? -- I think not req
        default_net_type: resnet_50??

4- Make subdirs that are copy of parent dir with suffix _AL_unif{}, where {}=n frames from active learning

5 - Run create_training_dataset in every subdir
    - copy h5 file and csv from training-datasets_
    - delete training-datasets_


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
# import random

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset



###################################
# %%
# Inputs
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


reference_dir_path = '/home/sofia/datasets/Horse10_AL2/Horses-Byron-2019-05-08'
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_AL2/horses_AL_train_test_idcs_split.pkl'

model_subdir_suffix = '_AL_unif{0:0=3d}' # subdirs with suffix _AL_unif{}, where {}=n frames from active learning
list_n_AL_frames = [0,10,50,100,500] # number of frames to sample from AL test set and pass to train set

###########################################
###########################################################
# %%
# Load train/test indices
with open(path_to_pickle_w_base_idcs,'rb') as file:
    # pickle.load(file)
    [map_shuffle_id_to_train_idcs,
     map_shuffle_id_to_test_AL_idcs,
     map_shuffle_id_to_test_OOD_idcs]=pickle.load(file)

##############################################
###########################################################
# %%
## Create  dir structure for each model
# create subdir for this model

# list of models = list of number of active learning frames to add to train set (& remove from test set)
for n_AL_samples in list_n_AL_frames:
    model_dir_path = reference_dir_path + model_subdir_suffix.format(n_AL_samples)
    shutil.copytree(reference_dir_path, #os.path.join(reference_dir_path,'training-datasets_'), 
                    model_dir_path) #os.path.join(model_dir_path,'training-datasets_'))

####################################################
###########################################################
# %%
# Create training dataset for train_AL_frames = 0
# OJO! idcs refer to dataframe h5 file in training-datasets_!!! need to replace
NUM_SHUFFLES = len(map_shuffle_id_to_train_idcs.keys())

# list of models = list of number of active learning frames to add to train set (& remove from test set)
for n_AL_samples in list_n_AL_frames:

    print('Model with {} active learning frames sampled'.format(n_AL_samples))

    ## Get model dir path and model config
    model_dir_path = reference_dir_path + model_subdir_suffix.format(n_AL_samples)
    config_path = os.path.join(model_dir_path,'config.yaml') 

    ###########################################################
    ## Create list of train and test idcs per shuffle for this model (list of lists ojo)
    train_idcs_one_model = []
    test_idcs_one_model = []
    for sh in range(1,NUM_SHUFFLES+1):
        # get base list of train and test_AL idcs for this shuffle
        list_base_train_idcs = map_shuffle_id_to_train_idcs[sh] 
        list_test_AL_idcs = map_shuffle_id_to_test_AL_idcs[sh]

        # compute indices of test indices to transfer to train set
        idx_test_AL_frames_to_transfer = [int(l) for l in np.floor(np.linspace(0,len(list_test_AL_idcs)-1,
                                                                    n_AL_samples))]

        # compute final lists of train and test indices for this shuffle, after transfer
        list_final_test_idcs = list_test_AL_idcs.copy() # we will pop from here
        list_test_AL_idcs_to_transfer = []  # to transfer to train set
        for el in sorted(idx_test_AL_frames_to_transfer,reverse=True): # we do reverse order to not alter indexing for next elements after poping
            list_test_AL_idcs_to_transfer.append(list_final_test_idcs.pop(el)) # pop by index
        list_final_train_idcs = list_base_train_idcs + list_test_AL_idcs_to_transfer
        
        # append results to lists of lists
        train_idcs_one_model.append(list_final_train_idcs)
        test_idcs_one_model.append(list_final_test_idcs)

    ###########################################################
    ## Create training dataset for this model (all shuffles)
    create_training_dataset(config_path,
                            num_shuffles=NUM_SHUFFLES,
                            trainIndices=train_idcs_one_model,  #map_shuffle_id_to_train_idcs[sh],
                            testIndices=test_idcs_one_model, # passing IID test indices for now
                            posecfg_template=pose_cfg_yaml_adam_path) # augmenter_type=None, posecfg_template=None,

    ###########################################################
    ## Copy h5 file from 'training-datasets_' to newly created 'training-datasets'                                  
    shutil.copyfile(os.path.join(model_dir_path,'training-datasets_',
                                'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'),
                    os.path.join(model_dir_path,'training-datasets',
                                'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'))

    ## Delete training-datasets_ directory (?)
    shutil.rmtree(os.path.join(model_dir_path,'training-datasets_'))


# %%
#########################################
# Check AL000
df = pd.read_hdf('/home/sofia/datasets/Horse10_AL2/Horses-Byron-2019-05-08/training-datasets_/iteration-0/'+\
                'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')

path_to_shuffle_pickle = '/home/sofia/datasets/Horse10_AL2/Horses-Byron-2019-05-08_AL000/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle3.pickle'

with open(path_to_shuffle_pickle, "rb") as f:
    pickledata = pickle.load(f)
    data = pickledata[0] 
    train_idcs = pickledata[1] 
    test_idcs = pickledata[2] 
    split_fraction = pickledata[3] # 

print(len(train_idcs)) #ok
print(len(test_idcs)) #ok
print('-----')
list_horses_in_train = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in train_idcs]))
list_horses_in_train.sort()
print(len(list_horses_in_train)) #ok
print(list_horses_in_train)
list_horses_in_test = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in test_idcs]))
list_horses_in_test.sort()
print(len(list_horses_in_test)) # ok
print(list_horses_in_test)

# %%
#########################################
# Check AL010
df = pd.read_hdf('/home/sofia/datasets/Horse10_AL2/Horses-Byron-2019-05-08/training-datasets_/iteration-0/'+\
                'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')

path_to_shuffle_pickle = '/home/sofia/datasets/Horse10_AL2/Horses-Byron-2019-05-08_AL010/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_53shuffle1.pickle'

with open(path_to_shuffle_pickle, "rb") as f:
    pickledata = pickle.load(f)
    data = pickledata[0] 
    train_idcs = pickledata[1] 
    test_idcs = pickledata[2] 
    split_fraction = pickledata[3] # 

print(len(train_idcs)) #ok
print(len(test_idcs)) #ok
print('-----')
list_horses_in_train = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in train_idcs]))
list_horses_in_train.sort()
print(len(list_horses_in_train)) #ok
print(list_horses_in_train)
list_horses_in_test = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in test_idcs]))
list_horses_in_test.sort()
print(len(list_horses_in_test)) # ok
print(list_horses_in_test)



# %%
####
# after copying                       
# l=list(set([df.iloc[el].name[1] for el in map_shuffle_id_to_train_idcs[1]]))
# l.sort()

# userfeedback=False,
# net_type='resnet_50',
# Shuffles=None,
# windows2linux=False,
# userfeedback=False,
# trainIndices=None,
# testIndices=None,
# net_type=None,
# augmenter_type=None,
# posecfg_template=None,