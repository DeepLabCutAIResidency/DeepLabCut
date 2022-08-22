#%%%
import os
import re
import shutil
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
#%%%

project_folder = "/home/jonas2/DLC_files/projects/"

#deeplabcut.create_new_project(
#            project='geneva_protocol_paper_austin_2020_bat_data',
#            experimenter='DLC',
#            videos=['/home/jonas2/DLC_files/projects/dummyVideos/'],
#            working_directory=project_folder
#        )

#%%%
config_path = "/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/config.yaml"
#%%%
#deeplabcut.convertcsv2h5(config_path, scorer= 'DLC', userfeedback=False)

# %%
#deeplabcut.check_labels(config_path)
# %%
# Dummy training dataset to get indexes and so on from later
#deeplabcut.create_training_dataset(config_path, Shuffles=[99])

#%%
import pandas as pd
df = pd.read_hdf('/home/jonas2/DLC_files/projects/geneva_protocol_paper_austin_2020_bat_data-DLC-2022-08-03/training-datasets/iteration-0/UnaugmentedDataSet_geneva_protocol_paper_austin_2020_bat_dataAug3/CollectedData_DLC.h5')
image_paths = df.index.to_list()


# %%
# train on A, shuffle 13
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[13],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %%
# train on A+25, shuffle 14
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_ref"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[14],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B, shuffle 15
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[15],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B+25, shuffle 16
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_ref"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[16],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %%
# train on A, shuffle 17
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[17],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %%
# train on A+25, shuffle 18
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_ref"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[18],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B, shuffle 19
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[19],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)
# %%
# train on A+B+25, shuffle 20
test_inds = []
train_inds = []
for i, path in enumerate(image_paths):
    if str(path[1]).endswith("_A"):
        train_inds.append(i)
    elif str(path[1]).endswith("_B"):
        train_inds.append(i)
    elif str(path[1]).endswith("_25_ref"):
        train_inds.append(i)
    elif str(path[1]).endswith("_50_test"):
        test_inds.append(i)

deeplabcut.create_training_dataset(
    config_path,
    Shuffles=[20],
    trainIndices=[train_inds],
    testIndices=[test_inds],
    net_type="resnet_101",
    augmenter_type="imgaug"
)

# %% ##################################################################################




# %% Create model folders

model_number = 1
modelprefix_pre = 'data_augm'
daug_str = 'fliplr'
TRAINING_SET_INDEX = 0
TRAIN_ITERATION = 0

#%%
### Get config as dict and associated paths
cfg = read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
training_datasets_path = os.path.join(project_path, "training-datasets")
#%%
# Get shuffles
shuffles = [17, 18, 19, 20]
trainingsetindices = [0, 1, 2, 3]
# %%
# Get train and test pose config file paths from base project, for each shuffle
list_base_train_pose_config_file_paths = []
list_base_test_pose_config_file_paths = []
for shuffle_number, trainingsetindex in zip(shuffles, trainingsetindices):
    base_train_pose_config_file_path_TEMP,\
    base_test_pose_config_file_path_TEMP,\
    _ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=shuffle_number,
                                            trainingsetindex=trainingsetindex)  # base_train_pose_config_file
    list_base_train_pose_config_file_paths.append(base_train_pose_config_file_path_TEMP)
    list_base_test_pose_config_file_paths.append(base_test_pose_config_file_path_TEMP)
# %%
# Create subdirs for this augmentation method
model_prefix = '_'.join([modelprefix_pre, "{0:0=2d}".format(model_number), daug_str]) # modelprefix_pre = aug_
aug_project_path = os.path.join(project_path, model_prefix)
aug_dlc_models = os.path.join(aug_project_path, "dlc-models", )
aug_training_datasets = os.path.join(aug_project_path, "training-datasets")
try:
    os.mkdir(aug_project_path)
except OSError as error:
    print(error)
    print("Skipping this one as it already exists")

# %% Copy training datasets in there
shutil.copytree(training_datasets_path, aug_training_datasets)

# %%
# Copy base train pose config file to the directory of this augmentation method
list_train_pose_config_path_per_shuffle = []
list_test_pose_config_path_per_shuffle = []
for j, (shuffle, trainingsetindex) in enumerate(zip(shuffles,trainingsetindices)):
    one_train_pose_config_file_path,\
    one_test_pose_config_file_path,\
    _ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=shuffle,
                                            trainingsetindex=trainingsetindex, # default
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
# %%
edits_dict = dict()
edits_dict["rotation"] = 90
edits_dict["gaussian_blur"] = False
edits_dict["gaussian_blur_params"] = {"sigma": (0.5, 4.0)}
edits_dict["scale_jitter_lo"] = .5
edits_dict["scale_jitter_up"] = 1.25
edits_dict["symmetric_pairs"] = (0, 14), (1, 12), (2, 13), (3, 11), (4, 9), (5, 10)
edits_dict["fliplr"] = True

## Initialise dict with additional edits to train config: optimizer
train_edits_dict = {}
dict_optimizer = {'optimizer':'adam',
    'batch_size': 8,
    'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 250000]]} # if no yaml file passed, initialise as an empty dict
train_edits_dict.update({'optimizer': dict_optimizer['optimizer'], #'adam',
    'batch_size': dict_optimizer['batch_size'], #16,
    'multi_step': dict_optimizer['multi_step']}) # learning rate schedule for adam: [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]

# %%
for j, shuffle in enumerate(shuffles):    
    edit_config(str(list_train_pose_config_path_per_shuffle[j]), edits_dict)
    edit_config(str(list_train_pose_config_path_per_shuffle[j]),
                        train_edits_dict)
    if 13 <= shuffle <= 20:
        edit_config(str(list_train_pose_config_path_per_shuffle[j]),
                {'intermediate_supervision': False})
        edit_config(str(list_train_pose_config_path_per_shuffle[j]),
                {'multi_stage': False})


    #elif 9 <= shuffle <= 12:
    #    edit_config(str(list_train_pose_config_path_per_shuffle[j]),
    #            {'intermediate_supervision': False})
    #    edit_config(str(list_train_pose_config_path_per_shuffle[j]),
    #            {'multi_stage': True})


# %%
