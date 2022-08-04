#%%
import os, shutil, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset
import re
import argparse
import yaml
#%%
model_number = 0
modelprefix_pre = 'data_augm'
daug_str = 'none'
config_path = '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/config.yaml'
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

list_shuffle_numbers = [1,2,3,4]
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
    shutil.copyfile(list_base_train_pose_config_file_paths[sh-1],
                        one_train_pose_config_file_path) 
    # copy base test config file
    shutil.copyfile(list_base_test_pose_config_file_paths[sh-1],
                        one_test_pose_config_file_path)
   # add to list
    list_train_pose_config_path_per_shuffle.append(one_train_pose_config_file_path) 
    list_test_pose_config_path_per_shuffle.append(one_test_pose_config_file_path)
# %%
edits_dict = dict()
edits_dict["rotation"] = 25
edits_dict["gaussian_blur"] = False
edits_dict["gaussian_blur_params"] = {"sigma": (0.5, 4.0)}
edits_dict["scale_jitter_lo"] = .5
edits_dict["scale_jitter_up"] = 1.25
edits_dict["symmetric_pairs"] = (0, 14), (1, 12), (2, 13), (3, 11), (4, 9), (5, 10)
edits_dict["fliplr"] = False



#%%
list_train_pose_config_path_per_shuffle = [
    '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/data_augm_00_none/dlc-models/iteration-0/geneva_protocol_paper_austin_2020_bat_dataJul29-trainset80shuffle1/train/pose_cfg.yaml',
    '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/data_augm_00_none/dlc-models/iteration-0/geneva_protocol_paper_austin_2020_bat_dataJul29-trainset84shuffle2/train/pose_cfg.yaml',
    '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/data_augm_00_none/dlc-models/iteration-0/geneva_protocol_paper_austin_2020_bat_dataJul29-trainset88shuffle3/train/pose_cfg.yaml',
    '/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-07-29/data_augm_00_none/dlc-models/iteration-0/geneva_protocol_paper_austin_2020_bat_dataJul29-trainset89shuffle4/train/pose_cfg.yaml'
]
# %%
for j, sh in enumerate(list_shuffle_numbers):
    edit_config(str(list_train_pose_config_path_per_shuffle[j]), edits_dict)
    edit_config(str(list_train_pose_config_path_per_shuffle[j]),
            {'fliplr': False,
            'symmetric_pairs': [(0,14), 
                                  (1,12),
                                  (2,13),
                                  (3,11),
                                  (4,9),
                                  (5,10)]})
# %%
## Initialise dict with additional edits to train config: optimizer
train_edits_dict = {}
dict_optimizer = {'optimizer':'adam',
    'batch_size': 8,
    'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 400000]]} # if no yaml file passed, initialise as an empty dict
train_edits_dict.update({'optimizer': dict_optimizer['optimizer'], #'adam',
    'batch_size': dict_optimizer['batch_size'], #16,
    'multi_step': dict_optimizer['multi_step']}) # learning rate schedule for adam: [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]
for j, sh in enumerate(list_shuffle_numbers):
    edit_config(str(list_train_pose_config_path_per_shuffle[j]),
                        train_edits_dict)
# %%
help(deeplabcut.train_network)
# %%
