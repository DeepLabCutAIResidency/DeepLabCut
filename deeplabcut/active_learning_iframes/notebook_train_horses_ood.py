# %%
import os, sys
import re 
import argparse
import yaml
import deeplabcut

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

# %%
######################################################
### Set config path
config_path = '/home/sofia/datasets/Horse10_ood copy/Horses-Byron-2019-05-08/config.yaml' 

## Set other params
NUM_SHUFFLES=3

GPU_TO_USE = 0
TRAINING_SET_INDEX = 3 # default;
MAX_SNAPSHOTS = 10
DISPLAY_ITERS = 1000 # display loss every N iters; one iter processes one batch
MAX_ITERS = 200_000
SAVE_ITERS = 50000 # save snapshots every n iters
TRAIN_ITERATION = 0 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

# %%
#####################################################
# Create training dataset---do not overwrite!
create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES,
    userfeedback=True) # augmenter_type=None, posecfg_template=None,

# %%
#############################################
# Edit train config

# GPU growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Adam
train_edits_dict = {'optimizer': 'adam', #'adam',
                    'batch_size': 8, #16,
                    'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]}

# get path to train config for each shuffle
for sh in range(NUM_SHUFFLES):
    one_train_pose_config_file_path,\
    _,_ = deeplabcut.return_train_network_path(config_path,
                                                shuffle=sh,
                                                trainingsetindex=TRAINING_SET_INDEX, 
                                                ) # modelprefix=modelprefix
    # add changes 
    edit_config(str(one_train_pose_config_file_path), 
                train_edits_dict)

# %%
#############################################
# Train
## Train each shuffle
for sh in range(NUM_SHUFFLES):
    deeplabcut.train_network(config_path, # config.yaml, common to all models
                                shuffle=sh,
                                trainingsetindex=TRAINING_SET_INDEX,
                                max_snapshots_to_keep=MAX_SNAPSHOTS,
                                displayiters=DISPLAY_ITERS,
                                maxiters=MAX_ITERS,
                                saveiters=SAVE_ITERS,
                                gputouse=GPU_TO_USE,
                                allow_growth=True,) # modelprefix=modelprefix



# %%
#############################################
# Input