# %%
import os, sys
import re 
import argparse
import yaml
import deeplabcut
import pickle
import pandas as pd

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

######################################################
# %% Read data from each pickle from original OOD setting

# from h5 file
path_to_h5_file_orig = '/home/sofia/datasets/Horse10_OOD/Horses-Byron-2019-05-08/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
df = pd.read_hdf(path_to_h5_file_orig)


NUM_SHUFFLES=3
list_train_idcs_per_shuffle = []
list_test_idcs_per_shuffle = []
for sh in range(1,NUM_SHUFFLES+1):

    # read pickles from original!
    path_to_shuffle_pickle = '/home/sofia/datasets/Horse10_OOD/Horses-Byron-2019-05-08/training-datasets/'+ \
                             'iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle{}.pickle'\
                             .format(sh)

    # read train/test idcs from pickle
    with open(path_to_shuffle_pickle, "rb") as f:
        pickledata = pickle.load(f)
        
        data = pickledata[0] # num_train_images = len(raw_data) 4041 ----why doesnt it match other sizes?
        train_idcs = pickledata[1] # 4057
        test_idcs = pickledata[2] #  4057
        split_fraction = pickledata[3] # 
        
        # append to lists
        list_train_idcs_per_shuffle.append(train_idcs)
        list_test_idcs_per_shuffle.append(test_idcs)

    # Check if train idcs of shuffle x are all from 10 horses
    if type(df.iloc[0,:].name) is tuple:
        list_horses_train_shuffle1 = list(set([df.iloc[t,:].name[1] for t in train_idcs]))  # df.iloc[t,:].name[1] 
    else:
        list_horses_train_shuffle1 = list(set([df.iloc[t,:].name.split('/')[1] for t in train_idcs]))  # df.iloc[t,:].name[1] 
    list_horses_train_shuffle1.sort()
    print('Num horses in train idcs shuffle {} = {}'. format(sh,
                                                            len(list_horses_train_shuffle1)))#----shouldnt this be 10?

# print list of idcs
print(len(list_train_idcs_per_shuffle))
print(len(list_test_idcs_per_shuffle))

[len(x)/len(y) for (x,y) in zip(list_train_idcs_per_shuffle,
                                list_test_idcs_per_shuffle)]
# for sh in range(3):
#     plt.plot(list_train_idcs_per_shuffle[sh],'.',label='shuffle {}'.format(sh))
#     plt.scatter(range(0,len(list_test_idcs_per_shuffle[sh])),
#                 list_test_idcs_per_shuffle[sh],
#                 10,
#                 c='b',label='shuffle {}'.format(sh))
# plt.show()
# %%
# #####################################################################################################
# Create training dataset for OOD setting
# - use idcs from pickle
# - use adam template for pose_cfg

# cp -r /home/sofia/datasets/Horse10_OOD /home/sofia/datasets/Horse10_OOD_modif
# change TrainingFraction!

config_path = '/home/sofia/datasets/Horse10_OOD_modif/Horses-Byron-2019-05-08/config.yaml' 
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'

create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES,
    userfeedback=False,
    net_type='resnet_50',
    trainIndices=list_train_idcs_per_shuffle,
    testIndices=list_test_idcs_per_shuffle,
    posecfg_template=pose_cfg_yaml_adam_path,
    ) # augmenter_type=None, posecfg_template=None,

    # Shuffles=None,
    # windows2linux=False,
    # userfeedback=False,
    # trainIndices=None,
    # testIndices=None,
    # net_type=None,
    # augmenter_type=None,
    # posecfg_template=None,
###########################################
# %%
#############################################
# Check horses per shuffle in newly created dataset

# from h5 file
path_to_h5_file_modif = '/home/sofia/datasets/Horse10_OOD_modif/Horses-Byron-2019-05-08/training-datasets/iteration-0/'+\
                        'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
df = pd.read_hdf(path_to_h5_file_modif)


NUM_SHUFFLES=3
list_train_idcs_per_shuffle = []
list_test_idcs_per_shuffle = []
for sh in range(1,NUM_SHUFFLES+1):

    # read pickles from original!
    path_to_shuffle_pickle_modif = '/home/sofia/datasets/Horse10_OOD_modif/Horses-Byron-2019-05-08/training-datasets/'+ \
                                    'iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle{}.pickle'\
                                    .format(sh)

    # read train/test idcs from pickle
    with open(path_to_shuffle_pickle_modif, "rb") as f:
        pickledata = pickle.load(f)
        
        data = pickledata[0] # num_train_images = len(raw_data) 4041 ----why doesnt it match other sizes?
        train_idcs = pickledata[1] # 4057
        test_idcs = pickledata[2] #  4057
        split_fraction = pickledata[3] # 
        
        # append to lists
        list_train_idcs_per_shuffle.append([train_idcs])
        list_test_idcs_per_shuffle.append([test_idcs])

    # Check if train idcs of shuffle x are all from 10 horses
    if type(df.iloc[0,:].name) is tuple:
        list_horses_train_shuffle1 = list(set([df.iloc[t,:].name[1] for t in train_idcs]))  
    else:
        list_horses_train_shuffle1 = list(set([df.iloc[t,:].name.split('/')[1] for t in train_idcs]))   
    list_horses_train_shuffle1.sort()
    print('Num horses in train idcs shuffle {} = {}'. format(sh,
                                                            len(list_horses_train_shuffle1)))#----shouldnt this be 10?


# %%
