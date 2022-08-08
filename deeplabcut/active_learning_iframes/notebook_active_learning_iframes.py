"""
Launch a refinement training job using a dataset created from iframes

We need to
- start training with weights from snapshot from baseline shuffle 1
- Create new training dataset by selecting i-frames from shuffles 2 and 3

"""

# %%
#####################################
## Import deeplabcut
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

import os
import pandas as pd
import pickle
import numpy as np

# %%
#######################################
# Input params
# - config, iteration=0, 50/50 split, shuffle=1 

path_to_shuffle2_pickle = \
'/home/sofia/datasets/Horse10_ood/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle1.pickle'
#'/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/Documentation_data-stinkbugs_80shuffle0.pickle'
# config_path = '/home/sofia/datasets/Horses-Byron-2019-05-08/config.yaml' 

# baseline_parent_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08/data_augm_00_baseline'
# path_to_shuffle2_pickle = os.path.join(baseline_parent_dir,
#                                       'training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle2.pickle')
# labelled_data_shuffle_3 = \
#     os.path.join(baseline_parent_dir,
#                  '/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle3.pickle')

# 
# init_weights = \
#     os.path.join(baseline_parent_dir,'/dlc-models/iteration-0/HorsesMay8-trainset50shuffle1/train/snapshot-200000')

## Set other params
# NUM_SHUFFLES=3

##################################################################
# %%
### Data from shuffles 2 and 3
# df_pickle_shuffle_2 = pd.read_pickle(path_to_shuffle2_pickle)
# with open(path_to_shuffle2_pickle, "rb") as f:
#         assemblies = pickle.load(f)

with open(path_to_shuffle2_pickle, "rb") as f:
    pickledata = pickle.load(f)
    
    data = pickledata[0] # num_train_images = len(raw_data) 4041 ----why doesnt it match other sizes?
    train_idcs = pickledata[1] # 4057
    test_idcs = pickledata[2] #  4057
    split_fraction = pickledata[3] # 
    
# training_data[0]['image'] / 'size' / 'joints'
# %%
###############################################
# Check if train and test idcs of shuffle 1 are all from the same 10 horses
df = pd.read_hdf('/home/sofia/datasets/Horse10_ood/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')
    #'/home/sofia/datasets/Horses-Byron-2019-05-08/data_augm_00_baseline/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')

list_horses_train_shuffle1 = list(set([df.iloc[t,:].name[1] for t in train_idcs])) #list(set([df.iloc[t,:].name[1] for t in train_idcs]))
list_horses_train_shuffle1.sort()
print(len(list_horses_train_shuffle1))#----shouldnt this be 10?

list_horses_test_shuffle1 = list(set([df.iloc[t,:].name[1] for t in test_idcs]))
list_horses_test_shuffle1.sort()
print(len(list_horses_test_shuffle1))

#################################################
# %%
# Compare output from train idcs in df and train_data[0]['image']
# df = pd.read_hdf('/home/sofia/datasets/Horses-Byron-2019-05-08/data_augm_00_baseline/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')

# train_idcs_sorted = np.sort(train_idcs)
for j in range(len(train_idcs)):
    if df.iloc[train_idcs[j],:].name != data[j]['image'] and \
       df.iloc[test_idcs[j],:].name != data[j]['image'] :
        print('error: {} not the same as {}'.format(df.iloc[train_idcs[j],:].name,
                                                    data[j]['image']))
        # break
    # else:
    #     print('ok: {} same as {}'.format(df.iloc[train_idcs[j],:].name,
    #                                      training_data[j]['image']))



# %%
###############################################
# # Create training dataset
# # based on i-frames from (shuffles? randomly sample a fraction of i-frames?)   
create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES) # augmenter_type=None, posecfg_template=-----> pass baseline config?,



# %%
