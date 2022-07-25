#%%
import os, shutil, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset
import re
import argparse
import yaml
#%%
model_number = 1
modelprefix_pre = 'data_augm'
daug_str = 'key_point_mirror'
config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-19/config.yaml'
TRAINING_SET_INDEX = 0
TRAIN_ITERATION = 0

#%%
### Get config as dict and associated paths
cfg = read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
training_datasets_path = os.path.join(project_path, "training-datasets")
#%%
# Get shuffles
iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(TRAIN_ITERATION))
dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
files_in_dataset_top_folder = os.listdir(dataset_top_folder)
list_shuffle_numbers = []
for file in files_in_dataset_top_folder:
    if file.endswith(".mat") and \
        str(int(cfg['TrainingFraction'][TRAINING_SET_INDEX]*100))+'shuffle' in file: # get shuffles for this training fraction idx only!
        shuffleNum = int(re.findall('[0-9]+',file)[-1])
        list_shuffle_numbers.append(shuffleNum)
# make list unique! (in case there are several training fractions)
list_shuffle_numbers = list(set(list_shuffle_numbers))
list_shuffle_numbers.sort()

# Get train and test pose config file paths from base project, for each shuffle
list_base_train_pose_config_file_paths = []
list_base_test_pose_config_file_paths = []
for shuffle_number in list_shuffle_numbers:
    base_train_pose_config_file_path_TEMP,\
    base_test_pose_config_file_path_TEMP,\
    _ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=shuffle_number,
                                            trainingsetindex=0)  # base_train_pose_config_file
    list_base_train_pose_config_file_paths.append(base_train_pose_config_file_path_TEMP)
    list_base_test_pose_config_file_paths.append(base_test_pose_config_file_path_TEMP)

#%%

#only do first one as test

list_shuffle_numbers = [list_shuffle_numbers[0]]
list_shuffle_numbers

#%%
###########################################################
# Create subdirs for this augmentation method
model_prefix = '_'.join([modelprefix_pre, "{0:0=2d}".format(model_number), daug_str]) # modelprefix_pre = aug_
aug_project_path = os.path.join(project_path, model_prefix)
aug_dlc_models = os.path.join(aug_project_path, "dlc-models", )
aug_training_datasets = os.path.join(aug_project_path, "training-datasets")
#%%%
# create subdir for this model
try:
    os.mkdir(aug_project_path)
except OSError as error:
    print(error)
    print("Skipping this one as it already exists")

# copy tree 'training-datasets' of dlc project under subdir for the current model---copies training_dataset subdir
shutil.copytree(training_datasets_path, aug_training_datasets)
###########################################################
# Copy base train pose config file to the directory of this augmentation method
list_train_pose_config_path_per_shuffle = []
list_test_pose_config_path_per_shuffle = []
for j, sh in enumerate(list_shuffle_numbers):
    one_train_pose_config_file_path,\
    one_test_pose_config_file_path,\
    _ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=sh,
                                            trainingsetindex=TRAINING_SET_INDEX, # default
                                            modelprefix=model_prefix)
    # make train and test directories for this subdir
    os.makedirs(str(os.path.dirname(one_train_pose_config_file_path))) # create parentdir 'train'
    os.makedirs(str(os.path.dirname(one_test_pose_config_file_path))) # create parentdir 'test
    # copy test and train config from base project to this subdir
    # copy base train config file
    shutil.copyfile(list_base_train_pose_config_file_paths[j],
                        one_train_pose_config_file_path) 
    # copy base test config file
    shutil.copyfile(list_base_test_pose_config_file_paths[j],
                        one_test_pose_config_file_path)
   # add to list
    list_train_pose_config_path_per_shuffle.append(one_train_pose_config_file_path) 
    list_test_pose_config_path_per_shuffle.append(one_test_pose_config_file_path)
# %%
edits_dict = dict()
edits_dict['rotation'] = 90
edits_dict["gaussian_blur"] = True
edits_dict["gaussian_blur_params"] = {"sigma": (0.0, 3.0)}


#%%
list_train_pose_config_path_per_shuffle
# %%
edit_config(str(list_train_pose_config_path_per_shuffle[0]), edits_dict)
# %%
## Initialise dict with additional edits to train config: optimizer
train_edits_dict = {}
dict_optimizer = {'optimizer':'adam',
    'batch_size': 8,
    'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]} # if no yaml file passed, initialise as an empty dict
train_edits_dict.update({'optimizer': dict_optimizer['optimizer'], #'adam',
    'batch_size': dict_optimizer['batch_size'], #16,
    'multi_step': dict_optimizer['multi_step']}) # learning rate schedule for adam: [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]
edit_config(str(list_train_pose_config_path_per_shuffle[0]),
                        train_edits_dict)
# %%
