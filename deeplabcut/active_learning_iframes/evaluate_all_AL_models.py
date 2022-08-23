
import os, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import argparse
import yaml

#########################################
## Input params
# TODO: add if __main__

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument("parent_dir_path", 
                    type=str,
                    help="path to parent directory [required]")
parser.add_argument("subdir_prefix_str", 
                    type=str,
                    help="prefix common to all subdirectories to train [required]")
parser.add_argument("gpu_to_use", 
                    type=int,
                    help="id of gpu to use (as given by nvidia-smi) [required]")
args = parser.parse_args()

#################################################################
# get required parameters
parent_dir_path = args.parent_dir_path #'/home/sofia/datasets/Horse10_AL_unif_OH'
model_prefix = args.subdir_prefix_str #'Horse10_AL_unif000'
GPU_TO_USE = args.gpu_to_use

TRAIN_ITERATION = 0 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

########################################
## Set 'allow growth' before eval (allow growth bug)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


##############################################
## Eval all shuffles per model
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
    dict_training_fraction_idx_per_shuffle = {}   
    for sh in list_shuffle_numbers:
        dict_training_fraction_per_shuffle[sh] =[float(re.search('_([0-9]*)shuffle{}.pickle'.format(sh), el).group(1))/100
                                                 for el in list_pickle_files_in_dataset_top_folder
                                                 if 'shuffle{}.pickle'.format(sh) in el][0]
        dict_training_fraction_idx_per_shuffle[sh] = \
            list_train_fractions_from_config.index(dict_training_fraction_per_shuffle[sh])

    ######################################
    # Train every shuffle
    for sh in list_shuffle_numbers: 
        deeplabcut.evaluate_network(config_path_one_model, # config.yaml, common to all models
                                    Shuffles=[sh],
                                    trainingsetindex=dict_training_fraction_idx_per_shuffle[sh], # index in list of training fraction,
                                    gputouse=GPU_TO_USE,
                                    modelprefix='') # ATT model prefix not passed directly, we are treating each model as a separate project (bc they have different config files)

#############################################
# config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
# NUM_SHUFFLES = 3
# TRAINING_SET_INDEX = 0
# GPU_TO_USE = 3
# MODEL_PREFIX = ''

# for sh in range(3):
#     deeplabcut.evaluate_network(config_path, # config.yaml, common to all models
#                                 Shuffles=[sh],
#                                 trainingsetindex=TRAINING_SET_INDEX,
#                                 gputouse=GPU_TO_USE,
#                                 modelprefix=MODEL_PREFIX)


#---
## Train this shuffle
    # deeplabcut.train_network(config_path_one_model, 
    #                             shuffle=sh,
    #                             trainingsetindex=dict_training_fraction_idx_per_shuffle[sh], # index in list of training fraction
    #                             max_snapshots_to_keep=MAX_SNAPSHOTS,
    #                             displayiters=DISPLAY_ITERS,
    #                             maxiters=MAX_ITERS,
    #                             saveiters=SAVE_ITERS,
    #                             gputouse=GPU_TO_USE,
    #                             allow_growth=True)