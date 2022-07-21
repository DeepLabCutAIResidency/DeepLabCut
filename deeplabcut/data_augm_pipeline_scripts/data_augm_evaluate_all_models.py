import os
from evaluate_all_shuffles import evaluate_all_shuffles
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config
import re
import argparse

def evaluate_all_shuffles(config_path, # config.yaml, common to all models
                          trainingsetindex=0,
                          gputouse=0,
                          modelprefix="",
                          train_iteration=0):

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

    ##############################################################
    ### Get list of shuffles for this model
    iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(train_iteration))
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

    ##########################################################
    ### Train every shuffle for this model
    for sh in list_shuffle_numbers:
        try: 
            deeplabcut.evaluate_network(config_path, # config.yaml, common to all models
                                        Shuffles=[sh],
                                        trainingsetindex=trainingsetindex,
                                        gputouse=gputouse,
                                        modelprefix=modelprefix)
        except FileNotFoundError:
            print('FileNotFound for model {}, shuffle {}, SKIPPING'.format(sh,modelprefix))
            continue

##################################################
if __name__ == "__main__":

    ##############################################################
    # ## Get command line input parameters
    # if an optional argument isn’t specified, it gets the None value (and None fails the truth test in an if statement)
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("config_path", 
                        type=str,
                        help="path to config.yaml file [required]")
    parser.add_argument("subdir_prefix_str", 
                        type=str,
                        help="prefix common to all subdirectories to train [required]")
    parser.add_argument("gpu_to_use", 
                        type=int,
                        help="id of gpu to use (as given by nvidia-smi) [required]")

    # Other  training params [optional]
    parser.add_argument("--training_set_index", 
                        type=int,
                        default=0,
                        help="Integer specifying which TrainingsetFraction to use. Note that TrainingFraction is a list in config.yaml.[optional]")
    parser.add_argument("--train_iteration", 
                        type=int,
                        default=0, # default is 0, but in stinkbug is 1. can this be extracted?
                        help="iteration number in terms of frames extraction and retraining workflow [optional]")
    args = parser.parse_args()

    ##############################################################
    ### Extract required input params
    config_path = args.config_path #"/media/data/stinkbugs-DLC-2022-07-15/config.yaml"
    subdir_prefix_str = args.subdir_prefix_str #str(sys.argv[2]) # "data_augm_"
    gpu_to_use = args.gpu_to_use #int(sys.argv[5])

    TRAINING_SET_INDEX = args.training_set_index # default;
    TRAIN_ITERATION = args.train_iteration # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

    ###########################################################################
    ## Compute list of subdirectories that start with 'subdir_prefix_str'
    list_all_dirs_in_project = os.listdir(str(os.path.dirname(config_path)))
    list_models_subdir = []
    for directory in list_all_dirs_in_project:
        if directory.startswith(subdir_prefix_str):
            list_models_subdir.append(directory)
    list_models_subdir.sort() # sorts in place

    #######################################################################
    ## Set 'allow growth' before evaluating? (allow growth bug)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    ###################################################################
    ## Evaluate models in list 
    for modelprefix in list_models_subdir:
        evaluate_all_shuffles(config_path, # config.yaml, common to all models
                             trainingsetindex=TRAINING_SET_INDEX, #0,
                             gputouse=gpu_to_use,
                             modelprefix=modelprefix,
                             train_iteration=TRAIN_ITERATION) #1)