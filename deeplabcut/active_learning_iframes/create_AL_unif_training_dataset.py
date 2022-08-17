'''
- This script prepares the training datasets for the active learning baseline, which
assumes samples are uniformly tranferred from the AL test set to the training set.

- We create N datasets to train N models, in each of them we sample uniformly x% of frames from the AL test set and add them to the train set,
for x in list_fraction_AL_frames

- The test idcs passed per shuffle are the frames in the testOOD set

The initial sets of train and AL test indices are loaded from the pickle file 'path_to_pickle_w_base_idcs'

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



###################################
# %%
# Inputs
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


reference_dir_path = '/home/sofia/datasets/Horse10_AL_unif_fr' #Horses-Byron-2019-05-08'#-------------
path_to_pickle_w_base_idcs = '/home/sofia/datasets/horses_AL_train_test_idcs_split.pkl'

model_subdir_prefix = 'Horse10_AL_unif{0:0=3d}' # subdirs with suffix _AL_unif{}, where {}=n frames from active learning
list_fraction_AL_frames = [25, 50, 75, 100] #[0,10,50,100,500] # number of frames to sample from AL test set and pass to train set

flag_pass_OOD_idcs_as_test_idcs = True # recommended: True (for a constant test set)

###########################################################
###########################################################
# %%
# Load train/test base indices from pickle
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

# rename training-datasets folder?

# list of models = list of number of active learning frames to add to train set (& remove from test set)
for fr_AL_samples in list_fraction_AL_frames:
    model_dir_path = os.path.join(reference_dir_path, model_subdir_prefix.format(fr_AL_samples))
    # copy labeled-data tree
    shutil.copytree(os.path.join(reference_dir_path,'labeled-data'), #os.path.join(reference_dir_path,'training-datasets_'), 
                    os.path.join(model_dir_path,'labeled-data'))  #os.path.join(model_dir_path,'training-datasets_'))
    # copy config
    shutil.copyfile(os.path.join(reference_dir_path,'config.yaml'),
                    os.path.join(model_dir_path,'config.yaml'))

####################################################
###########################################################
# %%
# Create training dataset for train_AL_frames = 0
# OJO! idcs refer to dataframe h5 file in training-datasets_!!! need to replace
NUM_SHUFFLES = len(map_shuffle_id_to_train_idcs.keys())

# list of models = list of number of active learning frames to add to train set (& remove from test set)
for fr_AL_samples in list_fraction_AL_frames:

    print('Model with {}%% active learning frames sampled'.format(fr_AL_samples))

    ## Get model dir path and model config
    model_dir_path = os.path.join(reference_dir_path, 
                                  model_subdir_prefix.format(fr_AL_samples)) #model_dir_path = reference_dir_path + model_subdir_preffix.format(n_AL_samples)
    config_path = os.path.join(model_dir_path,'config.yaml') 

    ###########################################################
    ## Create list of train and test idcs per shuffle for this model (list of lists ojo)
    train_idcs_one_model = []
    test_idcs_one_model = []
    list_training_fraction_per_shuffle = []
    for sh in range(1,NUM_SHUFFLES+1):
        # get base list of train and test_AL idcs for this shuffle
        list_base_train_idcs = map_shuffle_id_to_train_idcs[sh] 
        list_test_AL_idcs = map_shuffle_id_to_test_AL_idcs[sh]

        # compute indices of test indices to transfer to train set
        n_AL_samples = math.floor(fr_AL_samples*len(list_test_AL_idcs)/100)
        idx_test_AL_frames_to_transfer = [int(l) for l in np.floor(np.linspace(0,len(list_test_AL_idcs)-1,
                                                                    n_AL_samples))]

        # compute final lists of train and test indices for this shuffle, after transfer
        list_test_AL_idcs_wo_popped = list_test_AL_idcs.copy() # we will pop from here
        list_test_AL_idcs_to_transfer = []  # to transfer to train set
        for el in sorted(idx_test_AL_frames_to_transfer,reverse=True): # we do reverse order to not alter indexing for next elements after poping
            list_test_AL_idcs_to_transfer.append(list_test_AL_idcs_wo_popped.pop(el)) # pop by index
        list_final_train_idcs = list_base_train_idcs + list_test_AL_idcs_to_transfer
        
        # append results to lists of lists
        train_idcs_one_model.append(list_final_train_idcs)

        # select whether to pass remaining AL test idcs or OOD test idcs as 'final' test idcs
        if flag_pass_OOD_idcs_as_test_idcs:
            list_final_test_idcs = map_shuffle_id_to_test_OOD_idcs[sh]
        else:
            list_final_test_idcs = list_test_AL_idcs_wo_popped
        test_idcs_one_model.append(list_final_test_idcs)

        # add training fraction to list
        print('Shuffle {} train idcs: {}'.format(sh,len(list_final_train_idcs)))
        print('Shuffle {} test idcs: {}'.format(sh,len(list_final_test_idcs)))
        training_fraction_one_shuffle = \
            round(len(list_final_train_idcs) * 1.0 / (len(list_final_train_idcs)+len(list_final_test_idcs)), 2)
        training_fraction_one_shuffle = int(100*training_fraction_one_shuffle)/100 # to match name of shuffle saved!
                                                # int((len(list_final_train_idcs)/\
                                                #   (len(list_final_train_idcs)+len(list_final_test_idcs)))*100)/100
        list_training_fraction_per_shuffle.append(training_fraction_one_shuffle)

    ############################################################
    ## Edit training fraction in model's config
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
    ## Copy h5 file from 'training-datasets' at reference dir, to newly created 'training-datasets'  
    # it is overwritten after create_training_dataset() so I need to get it back from reference parent dir !   
    for ext in ['.csv','.h5']:                         
        shutil.copyfile(os.path.join(reference_dir_path,'training-datasets',
                                    'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron{}'.format(ext)), # src
                        os.path.join(model_dir_path,'training-datasets',
                                    'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron{}'.format(ext))) # dest

    ## Delete training-datasets_ directory (?)
    # shutil.rmtree(os.path.join(model_dir_path,'training-datasets_'))


# %%
#########################################
# Check AL000
df = pd.read_hdf(os.path.join(reference_dir_path,'training-datasets',
                             'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'))

path_to_shuffle_pickle = '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_53shuffle1.pickle' #53-1, 54-2, 50-3

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
df = pd.read_hdf(os.path.join(reference_dir_path,'training-datasets',
                             'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'))

path_to_shuffle_pickle = '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif010/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_53shuffle1.pickle' #53-1, 54-2, 50-3

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
