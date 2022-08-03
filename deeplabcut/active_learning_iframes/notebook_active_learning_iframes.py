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


# %%
#######################################
# Input params
# - config, iteration=0, 50/50 split, shuffle=1 

baseline_parent_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08/data_augm_00_baseline'
init_weights = \
    os.path.join(baseline_parent_dir,'/dlc-models/iteration-0/HorsesMay8-trainset50shuffle1/train/snapshot-200000')


#####################################################
# Create training dataset with shufflles 2 and 3

### Set config path
config_path = '/home/sofia/datasets/Horses-Byron-2019-05-08/config.yaml' 


### Data from shuffles 2 and 3
labelled_data_shuffle_2 = \
    os.path.join(baseline_parent_dir,
                 '/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle2.pickle')

labelled_data_shuffle_3 = \
    os.path.join(baseline_parent_dir,
                 '/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle3.pickle')

## Set other params
NUM_SHUFFLES=3
create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES) # augmenter_type=None, posecfg_template=None,

###############################################
# Create training dataset
# based on i-frames from (shuffles? randomly sample a fraction of i-frames?)   

