
import os, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import argparse
import yaml


#########################################
# Input params
parent_dir_path = '/home/sofia/datasets/Horse10_AL_unif_fr'
model_prefix = 'Horse10_AL_unif100'

MAX_SNAPSHOTS = 10
DISPLAY_ITERS = 1000 # display loss every N iters; one iter processes one batch
MAX_ITERS = 200_000
SAVE_ITERS = 50000 # save snapshots every n iters
TRAIN_ITERATION = 0 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

GPU_TO_USE=2

########################################
## Set 'allow growth' before training (allow growth bug)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#########################################
## Train all shuffles per model
list_models_dirs = [el for el in os.listdir(parent_dir_path) if el.startswith(model_prefix)]
list_models_dirs.sort()
for md in list_models_dirs:

    ### Get config path
    config_path_one_model = os.path.join(parent_dir_path, md,'config.yaml')
    cfg = read_config(config_path_one_model)
    list_train_fractions_from_config = cfg['TrainingFraction']

    ### Get list of shuffle numbers for this model
    training_datasets_path = os.path.join(parent_dir_path, md, "training-datasets")
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(TRAIN_ITERATION))
    dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
    list_pickle_files_in_dataset_top_folder = [el for el in os.listdir(dataset_top_folder)
                                                 if el.endswith('pickle')]

    list_shuffle_numbers = [int(re.search('shuffle(.*).pickle', el).group(1))
                            for el in list_pickle_files_in_dataset_top_folder]
    list_shuffle_numbers.sort()   

    ### Get list of training set **idcs** per shuffle
    # idcs follow list in config ojo
    dict_training_fraction_per_shuffle = {}
    dict_training_idx_per_shuffle = {}   
    for sh in list_shuffle_numbers:
        dict_training_fraction_per_shuffle[sh] =[float(re.search('_([0-9]*)shuffle{}.pickle'.format(sh), el).group(1))/100
                                                 for el in list_pickle_files_in_dataset_top_folder
                                                 if 'shuffle{}.pickle'.format(sh) in el][0]
        dict_training_idx_per_shuffle[sh] = \
            list_train_fractions_from_config.index(dict_training_fraction_per_shuffle[sh])

    ######################################
    # Train every shuffle
    for sh in list_shuffle_numbers: 

        ## Train this shuffle
        deeplabcut.train_network(config_path_one_model, 
                                 shuffle=sh,
                                 trainingsetindex=dict_training_idx_per_shuffle[sh],
                                 max_snapshots_to_keep=MAX_SNAPSHOTS,
                                 displayiters=DISPLAY_ITERS,
                                 maxiters=MAX_ITERS,
                                 saveiters=SAVE_ITERS,
                                 gputouse=GPU_TO_USE,
                                 allow_growth=True)