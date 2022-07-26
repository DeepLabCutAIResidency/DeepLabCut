"""
TODO
- all subdirectories that start with model prefix
- save results as pickle?
- refactor pck? save results as vars
"""


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import re

# Import 'local' deeplabcut ---- use relative path instead?
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


<<<<<<< Updated upstream
#########################################
# Input params
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'#'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
=======
##########################################
# Read files from evaluated network
human_labels_filepath ='/home/sabrina/Horses-Byron-2019-05-08/data_augm_00_baseline/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.csv' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
model_predictions_filepath = '/home/sabrina/Horses-Byron-2019-05-08/data_augm_00_baseline/evaluation-results/iteration-0/HorsesMay8-trainset5shuffle0/DLC_resnet50_HorsesMay8shuffle0_200000-snapshot-200000.h5'
# '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/evaluation-results/iteration-0/sabris-mouseJul6-trainset80shuffle1/DLC_resnet50_sabris-mouseJul6shuffle1_2-snapshot-2.h5'


# Read config of trained network
config_path = '/home/sabrina/Horses-Byron-2019-05-08/config.yaml'#'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
>>>>>>> Stashed changes
# NUM_SHUFFLES=1 # this is an input to create_training_dataset but I think it is not saved anywhere


TRAIN_ITERATION = 1
TRAINING_SET_INDEX = 0
SUBDIR_STR = 'data_augm_00_baseline'
thresh = 1

##########################################
# Compute paths to subdir
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
training_datasets_path = os.path.join(project_path, SUBDIR_STR,  "training-datasets")
unaugmented_training_dataset_path = os.path.join(SUBDIR_STR,
                                                 auxiliaryfunctions.GetTrainingSetFolder(cfg))
                                               

##########################################
# Read human labelled data for this project
iteration_folder = os.path.join(training_datasets_path, 'iteration-' + str(TRAIN_ITERATION))
dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
files_in_dataset_top_folder = os.listdir(dataset_top_folder)

h5_files_in_dataset_top_folder = \
        [el for el in files_in_dataset_top_folder if  el.endswith('h5')]
if len(h5_files_in_dataset_top_folder) > 1:
        print('More than one h5 file found at {}, selecting first one...'.format(dataset_top_folder))
human_labels_filepath = os.path.join(dataset_top_folder,
                                     h5_files_in_dataset_top_folder[0])

# '/media/data/stinkbugs-DLC-2022-07-15/data_augm_00_baseline/training-datasets/iteration-1/UnaugmentedDataSet_stinkbugsJul15/CollectedData_DLC.h5' #'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
# '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/evaluation-results/iteration-0/sabris-mouseJul6-trainset80shuffle1/DLC_resnet50_sabris-mouseJul6shuffle1_2-snapshot-2.h5'

df_human = pd.read_hdf(human_labels_filepath)

##############################################################
### Get list of shuffles for this model
list_training_fractions = cfg["TrainingFraction"]
list_shuffle_numbers = []
for file in files_in_dataset_top_folder:
        if file.endswith(".mat") and \
        str(int(list_training_fractions[TRAINING_SET_INDEX]*100))+'shuffle' in file: # get shuffles for this training fraction idx only!

                shuffleNum = int(re.findall('[0-9]+',file)[-1])
                list_shuffle_numbers.append(shuffleNum)
# make list unique! (in case there are several training fractions)
list_shuffle_numbers = list(set(list_shuffle_numbers))
list_shuffle_numbers.sort()

##############################################################
## Compute dataframe with error per keypoint for each shuffle
# Loop thru shuffles
dict_df_results_per_shuffle = dict()
for shuffle in list_shuffle_numbers: #[list_shuffle_numbers[0]]: #range(1,NUM_SHUFFLES+1):

        ### Get latest snapshot for this shuffle
        path_to_snapshot_parent_dir =\
                 auxiliaryfunctions.get_evaluation_folder(list_training_fractions[TRAINING_SET_INDEX],
                                                        shuffle, 
                                                        cfg, 
                                                        modelprefix=SUBDIR_STR)  
        
        list_snapshots = [el for el in os.listdir(os.path.join(project_path,
                                                               str(path_to_snapshot_parent_dir))) if el.endswith(".h5")]

        list_snapshot_iter = [int(re.findall('[0-9]+.h5',el)[0].split('.')[0]) for el in list_snapshots]
        idx_latest_snapshot = np.argmax(np.array(list_snapshot_iter))                                                     
        model_predictions_one_shuffle_filepath = \
                os.path.join(project_path,
                             str(path_to_snapshot_parent_dir),
                             list_snapshots[idx_latest_snapshot])

        ### Read model predictions
        df_model_one_shuffle = pd.read_hdf(model_predictions_one_shuffle_filepath)

        ### Get test indices for this shuffle
        _, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(unaugmented_training_dataset_path, 
                                                                       list_training_fractions[TRAINING_SET_INDEX], 
                                                                       shuffle, 
                                                                       cfg)
        _,_, testIndices, _ = auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"], 
                                                                           metadatafn))

        # Get rows from test set only
        df_human_test_only = df_human.iloc[testIndices,:]  # test idcs form images in ascending order?
        df_model_test_only = df_model_one_shuffle.iloc[testIndices,:]

        # Drop scorer level
        df_human_test_only = df_human_test_only.droplevel('scorer',axis=1)
        df_model_test_only = df_model_test_only.droplevel('scorer',axis=1)   

        ### Compute x and y error: deltas in x and y dir between human scorer and model prediction
        df_diff_test_only = df_human_test_only - df_model_test_only
        

        #### Compute distance btw model and human
        # - nrows = samples in test set
        # - ncols = bodyparts tracked
        # Drop llk for model predictions before computing distance
        df_diff_test_only = df_diff_test_only.drop(labels='likelihood',axis=1,level=1)
        df_distance_test_only = df_diff_test_only.pow(2).sum(level='bodyparts',axis=1,skipna=False).pow(0.5)
        # warning: recommends to use 'df_diff_test_only.pow(2).groupby(level='bodyparts',axis=1).sum(axis=1,skipna=False)' instead,
        # but that makes NaNs into 0s!
        # add distance level
        df_distance_test_only.columns = pd.MultiIndex.from_product([df_distance_test_only.columns, ['distance']])

        ## Combine w Likelihood
        df_llk_test_only = df_model_test_only.drop(labels=['x','y'],axis=1,level=1)
        df_results = pd.concat([df_distance_test_only,df_llk_test_only],axis=1).sort_index(level=0,axis=1)

        dict_df_results_per_shuffle.update({shuffle:df_results})


###########################################################
# Compute PCK per shuffle
for shuffle in list_shuffle_numbers:
        df_results  = dict_df_results_per_shuffle[shuffle]
        np_axes_per_row = df_results.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1)
        all = 0
        bdpts = list(np_axes_per_row.columns.values)

        # PCK per bodypart
        for bdpt in bdpts:
                val_per_bdpt = list(np_axes_per_row[bdpt])

                count = np_axes_per_row[bdpt][np_axes_per_row[bdpt] > thresh].count()
                # print('The number of keypoint above  ' +str( thresh) + ' is ' + str(count) )#+ ' in ' + str(len(val_per_bdpt)))
                all += count

        # PCK overall
        print('The total number of keypoint above  ' +str( thresh) + ' is ' + str(all))


##############################################################
# Mean and sigma per bodypart
dict_px_error_by_bodypart = dict()
dict_px_error_total = dict()
for shuffle in list_shuffle_numbers:
        df_results  = dict_df_results_per_shuffle[shuffle]
        df_summary_per_bodypart = df_results.describe()
        dict_px_error_by_bodypart.update({shuffle:df_summary_per_bodypart})
        print('----------------------')
        print('Shuffle {}, pixel error per bodypart:'.format(shuffle))
        print(df_summary_per_bodypart)
        

        # Mean and sigma across all bodyparts
        px_error_all_bodyparts_and_test_samples = np.nanmean(df_results.drop(labels='likelihood',axis=1,level=1)) # matches result for evaluate fn
        dict_px_error_total.update({shuffle:px_error_all_bodyparts_and_test_samples})
        print('----------------------')
        print('Shuffle {}, pixel error all bodyparts and test samples:'.format(shuffle))
        print(px_error_all_bodyparts_and_test_samples)
        print('----------------------')

######################################################
# Save results as pickle file?
# dict_df_results_per_shuffle
# dict_pck
# dict_px_error_by_bodypart = dict()
# dict_px_error_total = dict()



##############################################################
# Plot histogram per bodypart
# https://mode.com/example-gallery/python_histogram/
# drop 'distance' level column before plotting
np_axes_per_row = df_results.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1).hist()  # returns: matplotlib.AxesSubplot or numpy.ndarray of them
for ax in np_axes_per_row:
    for i in range(ax.shape[0]):
        # Title
        # ax[i].title
        # Despine
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

        # Switch off ticks
        ax[i].tick_params(axis='both',which='both',\
                        bottom='off', top='off',labelbottom='on')
        # Draw horiz lines per ytick
        for yt in ax[i].get_yticks():
                ax[i].axhline(y=yt, linestyle='-',alpha=0.4, color='#eeeeee', zorder=1)

        # Set x and y labels
        ax[i].set_xlabel('error-distance (pixels)', weight='bold', size=12)
        ax[i].set_ylabel('counts',weight='bold',size=12) # labelpad=5,
        
        # Format y-axis label
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax[i].tick_params(axis='x', rotation=0)

