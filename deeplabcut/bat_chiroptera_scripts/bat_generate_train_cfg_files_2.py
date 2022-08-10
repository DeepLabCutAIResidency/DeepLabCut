"""
This script generates a set of train config files (pose_config.yaml files) for a bat study
This is for the second set where I change the augmentation

Example usage (from DeepLabCut directory):
     python deeplabcut/data_augm_pipeline_scripts/data_augm_generate_train_cfg_files.py  '/media/data/stinkbugs-DLC-2022-07-15-SMALL/config.yaml' 'data_augm' 'deeplabcut/data_augm_pipeline_scripts/baseline.yaml' --train_iteration=1

Contributors: Sofia, Jonas, Sabrina
"""

import os, shutil
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re
import argparse
import yaml

def create_parameters_dict():
    ##################################################################
    ### Define parameters for each data augmentation method --maybe read a separate file?
    # ATT! Parameters must be defined for True or False cases.
    # Not defining a set of parameters will result in applying the parameters from the pose_config.yaml template

    ## Initialise baseline dict with params per data augm type
    parameters_dict = dict() # define a class instead of a dict?

    ### General
    parameters_dict['general'] = {'dataset_type': 'imgaug', # OJO! not all the following will be available?
                                    'batch_size': 1, # 128
                                    'apply_prob': 0.5,
                                    'pre_resize': []} # Specify [width, height] if pre-resizing is desired

    ### Crop----is this applied if we select imgaug? I think so....
    parameters_dict['crop'] = {False: {'crop_by': 0.0,
                                    'cropratio': 0.0},
                            True: {'crop_by': 0.15,
                                    'cropratio': 0.4}}#---------- these are only used if height, width passed to pose_imgaug
    # from template:
    # parameters_dict['crop'] = {'crop_size':[400, 400],  # width, height,
    #                   'max_shift': 0.4,
    #                   'crop_sampling': 'hybrid',
    #                   'cropratio': 0.4}---------- crop ratio is used too

    ### Rotation
    parameters_dict['rotation'] = {False:{'rotation': 0,
                                        'rotratio': 0},
                                    True:{'rotation': 25,
                                        'rotratio': 0.4}}

    ### Scale
    parameters_dict['scale'] = {False:{'scale_jitter_lo': 1.0,
                                    'scale_jitter_up': 1.0},
                                True:{'scale_jitter_lo': 0.5,
                                    'scale_jitter_up': 1.25}}


    ### Motion blur
    # ATT motion_blur is not expected as a dictionary
    parameters_dict['motion_blur'] = {False: {'motion_blur': False}, # motion_blur_params should not be defined if False, but check if ok
                                    True: {'motion_blur': True,
                                            'motion_blur_params':{"k": 7, "angle": (-90, 90)}}}

    ### Contrast
    # ATT for Contrast a dict should be defined in the yaml file!
    # also: log, linear, sigmoid, gamma params...include those too? [I think if they are not defined in the template we are good, they wont be set]
    parameters_dict['contrast'] = {False: {'contrast': {'clahe': False,
                                                        'histeq': False}}, # ratios should not be defined if False, but check if ok
                                    True:{'contrast': {'clahe': True,
                                                        'claheratio': 0.1,
                                                        'histeq': True,
                                                        'histeqratio': 0.1}}}


    ### Convolution
    # ATT for Convolution a dict should be defined in the yaml file!
    parameters_dict['convolution'] = {False: {'convolution': {'sharpen': False,  # ratios should not be defined if False, but check if ok
                                                            'edge': False,
                                                            'emboss': False}}, # this needs to be fixed in pose_cfg.yaml template?
                                    True: {'convolution':{'sharpen': True,
                                                            'sharpenratio': 0.3, #---- in template: 0.3, in pose_imgaug default is 0.1
                                                            'edge': True,
                                                            'edgeratio': 0.1, #--------
                                                            'emboss': True,
                                                            'embossratio': 0.1}}}
    ### Mirror
    parameters_dict['mirror'] = {False: {'mirror': False},
                                True: {'mirror': True}}

    ### Grayscale
    parameters_dict['grayscale'] = {False: {'grayscale': False},
                                    True: {'grayscale': True}}

    ### Covering
    parameters_dict["covering"] = {False: {'covering': False},
                                True: {'covering': True}}

    ### Elastic transform
    parameters_dict["elastic_transform"] = {False: {'elastic_transform': False},
                                            True: {'elastic_transform': True}}

    ### Gaussian noise
    parameters_dict['gaussian_noise'] = {False: {'gaussian_noise': False},
                                        True: {'gaussian_noise': True}}                                         

    ### Cloudy weather
    parameters_dict['cloudy'] = {False: {'clouds': False,
                                         'fog': False,
                                         'rain': False},
                                True: {'clouds': True,
                                         'fog': True,
                                         'rain': True}}
    ### Snowy weather
    parameters_dict['snowy'] = {False: {'snow': False,
                                        'snow_flakes': False},
                                True: {'snow': True,
                                        'snow_flakes': True}}


    return parameters_dict                                    


#############################################
if __name__ == "__main__":

    ##############################################################
    ## Parse command line input parameters
    # (we assume create_training_dataset has already been run)
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("config_path", #'/media/data/stinkbugs-DLC-2022-07-15/config.yaml' # '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
                        type=str,
                        help="path to config.yaml file [required]")
    parser.add_argument("subdir_prefix_str", 
                        type=str,
                        help="prefix common to all subdirectories to train [required]")
    parser.add_argument("baseline_yaml_file_path", 
                        type=str,
                        default='',
                        help="path to file that defines the data augmentation baseline [required]")                       
    # optional
    parser.add_argument('-l',"--list_data_augm_idcs_to_flip", 
                        type=int,
                        nargs='+',
                        default=[], #-----------------
                        help="List of indices of data augmentation methods (as listed in baseline yaml file) to inspect effect of.\
                              If no list is provided, the script generates a train config with every method 'flipped' \
                              wrt the baseline [optional]")
    parser.add_argument("--training_set_index", 
                        type=int,
                        default=0,
                        help="Integer specifying which TrainingsetFraction to use. Note that TrainingFraction is a list in config.yaml.[optional]")
    parser.add_argument("--train_iteration", 
                        type=int,
                        default=0, # default is 0, but in stinkbug is 1. can this be extracted?
                        help="iteration number in terms of frames extraction and retraining workflow [optional]")
    args = parser.parse_args()
    
    ##########################################################
    ### Extract required input params
    config_path = args.config_path
    # each model subfolder is named with the format: <modelprefix_pre>_<id>_<str_id>
    modelprefix_pre = args.subdir_prefix_str #"data_augm"
    baseline_yaml_file_path = args.baseline_yaml_file_path

    # Other params
    TRAINING_SET_INDEX = args.training_set_index # default;
    TRAIN_ITERATION = args.train_iteration # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?

    ##########################################################
    ### Get config as dict and associated paths
    cfg = read_config(config_path)
    project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
    training_datasets_path = os.path.join(project_path, "training-datasets")

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

    ###############################################################
    ## Create params dict -----potentially as a yaml file?
    parameters_dict = create_parameters_dict()

    ############################################################################
    ## Define baseline
    with open(args.baseline_yaml_file_path,'r') as yaml_file:
        baseline = yaml.safe_load(yaml_file)
    list_baseline_keys = list(baseline.keys())


    #################################################
    ## Create list of strings identifying each model
    # if required: consider only specific data augmentation methods

    list_of_data_augm_models_strs = ['baseline']

    #########################################
    ## Loop to train each model
    for i, daug_str in enumerate(list_of_data_augm_models_strs):
        ###########################################################
        # Create subdirs for this augmentation method
        model_prefix = '_'.join([modelprefix_pre, "{0:0=2d}".format(i), daug_str]) # modelprefix_pre = aug_
        aug_project_path = os.path.join(project_path, model_prefix)
        aug_dlc_models = os.path.join(aug_project_path, "dlc-models", )
        aug_training_datasets = os.path.join(aug_project_path, "training-datasets")
        # create subdir for this model
        try:
            os.mkdir(aug_project_path)
        except OSError as error:
            print(error)
            print("Skipping this one as it already exists")
            continue
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
            os.makedirs(str(os.path.dirname(one_test_pose_config_file_path))) # create parentdir 'test'

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

        #####################################################
        # Create dict with the data augm params for this model
        # initialise dict with gral params
        edits_dict = dict()
        edits_dict.update(parameters_dict['general'])
        for ky in list_baseline_keys: #baseline.keys():
            if daug_str == ky:
                # Get params that correspond to the opposite state of the method daug_str in the baseline
                d_temp = parameters_dict[ky][not baseline[ky]]
                # add to edits dict
                edits_dict.update(d_temp)
            else:
                # Get params that correspond to the same state as the baseline
                d_temp = parameters_dict[ky][baseline[ky]]
                # add to edits dict
                edits_dict.update(d_temp)

        # print
        print('-----------------------------------')
        if daug_str == 'baseline':
            print('Data augmentation model {}: {}'.format(i, daug_str))
        else:
            print('Data augmentation model {}: "{}" opposite to baseline'.format(i, daug_str))
        [print('{}: {}'.format(k,v)) for k,v in edits_dict.items()]
        print('-----------------------------------')

        ##################################################
        # Edit config for this data augmentation setting
        for j, sh in enumerate(list_shuffle_numbers):
            edit_config(str(list_train_pose_config_path_per_shuffle[j]), edits_dict)