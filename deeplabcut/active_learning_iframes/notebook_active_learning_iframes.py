"""
Launch a refinement training job using a dataset created from iframes

We need to
- start training with weights from snapshot from baseline shuffle 1
- Create new training dataset by selecting i-frames from shuffles 2 and 3

"""

# %%
#####################################
import deeplabcut




# %%
#######################################
# Input params
# - config, iteration=0, 50/50 split, shuffle=1 
init_weights = \
    '/home/sofia/datasets/Horses-Byron-2019-05-08/data_augm_00_baseline/dlc-models/iteration-0/HorsesMay8-trainset50shuffle1/train/snapshot-200000'

######################################
# Create training dataset
# based on i-frames from (shuffles? randomly sample a fraction of i-frames?)    