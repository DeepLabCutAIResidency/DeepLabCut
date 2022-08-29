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
    Here, the AL train frames are ranked based on the images influence. 
    
    
- The test idcs passed per shuffle are the frames in the testOOD set

- The sets of train indices (for both base and AL sets) and test indices per shuffle
  are loaded from the pickle file 'path_to_pickle_w_base_idcs'

- Before running this script:
    - Download horse10.tar.gz and extract in 'Horse10_AL_infl_OH': (use --strip-components 1)
        mkdir <reference_dir_path> (e.g. Horse10_AL_infl_OH)
        tar -xvzf horse10.tar.gz -C <reference_dir_path> --strip-components 1


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

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

#########################################################################################
# %%
### Inputs

# parent directory and base data
AL_strategy_str = 'AL_unif_small' # uniform sampling: AL_unif; AL_uncert; AL_infl
flag_reverse_sampled_idcs = False # whether to start sampling from the end of ranked list

reference_dir_path = f'/home/sofia/datasets/Horse10_{AL_strategy_str}_OH' 
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_test_idcs_split.pkl'

# models subdirectory prefix
model_subdir_prefix = 'Horse10_{}{:0=3d}' # subdirs with suffix _AL_unif{}, where {}=n frames from active learning
list_fraction_AL_frames = [0, 5, 10, 15, 20, 25] # [0,10,50,100,500] # number of frames to sample from AL test set and pass to train set

# path to train AL idcs ranked by X
# if empty string: frames are sampled uniformly
path_to_pickle_w_AL_train_idcs_ranked = '' #/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_uncert_kmeans_ranked_idcs.pkl'

# train config template (with adam params)
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'


###########################################################
# %%
# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [map_shuffle_id_to_base_train_idcs,
      map_shuffle_id_to_AL_train_idcs,
      map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)


#########################################################
# %% Check
if bool(path_to_pickle_w_AL_train_idcs_ranked == '') !=  bool('AL_unif' in AL_strategy_str):
    print('ERROR: there is a mismatch between the AL inputs. \
          Either a uniform strategy is selected but a pickle with ranked idcs is provided, or a \
          the pickle file path is empty but a strategy different from uniform is selected')  
    sys.exit()
elif (path_to_pickle_w_AL_train_idcs_ranked == '') and ('AL_unif' in AL_strategy_str):
    print('Active learning sampling: uniform across frames (ct step)')
###########################################################
# %%
## Create  dir structure for each AL model
# list of models = list of fractions of active learning frames to add to train set 
for fr_AL_samples in list_fraction_AL_frames:
    print(fr_AL_samples)
    model_dir_path = os.path.join(reference_dir_path, model_subdir_prefix.format(AL_strategy_str,fr_AL_samples))
    # copy labeled-data tree
    shutil.copytree(os.path.join(reference_dir_path,'labeled-data'),
                    os.path.join(model_dir_path,'labeled-data'))  
    # copy config
    shutil.copyfile(os.path.join(reference_dir_path,'config.yaml'),
                    os.path.join(model_dir_path,'config.yaml'))


###########################################################
# %% Get groundtruth and AL train samples ranked by X
df_groundtruth = pd.read_hdf(path_to_h5_file)
if path_to_pickle_w_AL_train_idcs_ranked != '':
    with open(path_to_pickle_w_AL_train_idcs_ranked,'rb') as file:
        print('Loading ranked AL train idcs from {}'.format(path_to_pickle_w_AL_train_idcs_ranked))
        map_shuffle_id_to_AL_train_idcs_ranked = pickle.load(file)

if flag_reverse_sampled_idcs:
    print('AL train idcs are sampled in reverse order')
#########################################################################################
# %%
# Create training datasets
# ATT! idcs refer to dataframe h5 file in training-datasets dir, as it is before running create_training_dataset!!! 
# So I need to replace it

# list of models = list of number of active learning frames to add to train set 
NUM_SHUFFLES = len(map_shuffle_id_to_base_train_idcs.keys())
for fr_AL_samples in list_fraction_AL_frames:
    print('------------------------------------------')
    print('Model with {}% of active learning frames sampled'.format(fr_AL_samples))

    ## Get model dir path and model config
    model_dir_path = os.path.join(reference_dir_path, 
                                  model_subdir_prefix.format(AL_strategy_str,fr_AL_samples)) 
    config_path = os.path.join(model_dir_path,'config.yaml') 

    ###########################################################
    ## Create list of train and test idcs per shuffle for this model (list of lists)
    train_idcs_one_model = []
    test_idcs_one_model = []
    list_training_fraction_per_shuffle = []
    for sh in range(1,NUM_SHUFFLES+1):
        # get list of base train for this shuffle
        list_base_train_idcs = map_shuffle_id_to_base_train_idcs[sh] 
        
        # sample AL idcs
        n_AL_samples = math.floor(fr_AL_samples*len(map_shuffle_id_to_AL_train_idcs[sh])/100)
        if bool(path_to_pickle_w_AL_train_idcs_ranked == ''): #AL_strategy_str=='AL_unif':
            #---------------------------------------------------------------------------
            # Compute indices of AL train indices to transfer to train set ---sample randomly instead?
            # n_AL_samples = math.floor(fr_AL_samples*len(list_AL_train_idcs)/100)
            idx_AL_train_idcs_to_transfer = \
                [int(l) for l in np.floor(np.linspace(0,len(map_shuffle_id_to_AL_train_idcs[sh])-1,
                                                      n_AL_samples, endpoint=False))] #endpoint=True makes last sample=stop value
            #---------------------------------------------------------------------------

            ## Compute final lists of train indices for this shuffle, after transfer
            list_AL_train_idcs = map_shuffle_id_to_AL_train_idcs[sh]
            list_AL_train_idcs_to_transfer = [list_AL_train_idcs[x] for x in idx_AL_train_idcs_to_transfer]
            # list_AL_train_idcs_wo_popped = list_AL_train_idcs.copy() # we will pop from here
            # list_AL_train_idcs_to_transfer = []  # to transfer to train set
            # for el in sorted(idx_AL_train_idcs_to_transfer,reverse=True): # we do reverse order to not alter indexing for next elements after poping
            #     list_AL_train_idcs_to_transfer.append(list_AL_train_idcs_wo_popped.pop(el)) # pop by index

        else:
            #---------------------------------------------------------------------------
            # get list of top x% AL train idcs, ranked by desired mpe metric
            # n_AL_samples = math.floor(fr_AL_samples*len(map_shuffle_id_to_AL_train_idcs_ranked[sh])/100)
            if flag_reverse_sampled_idcs:
                list_AL_train_idcs_to_transfer = map_shuffle_id_to_AL_train_idcs_ranked[sh][-n_AL_samples:]
            else:
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
                              model_subdir_prefix.format(AL_strategy_str,fr_AL_samples)) 
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
