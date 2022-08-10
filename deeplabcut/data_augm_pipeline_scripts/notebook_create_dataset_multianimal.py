## Import deeplabcut
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset


######################################################
### Set config path
config_path = '/media/data/trimice-dlc-2021-06-22_batchSize1/config.yaml' 

############################################################
## Set other params
NUM_SHUFFLES=3

#####################################################
# Create training dataset
create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES) # augmenter_type=None, posecfg_template=None,
