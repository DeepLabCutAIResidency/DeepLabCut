"""
TODO
- all subdirectories that start with model prefix
- save results as pickle or csv?
- refactor pck? save results as vars
"""

# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import re

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


#########################################
# Input params
config_path = '/home/sabrina/Horses-Byron-2019-05-08/config.yaml'#'/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
# NUM_SHUFFLES=1 # this is an input to create_training_dataset but I think it is not saved anywhere

TRAIN_ITERATION = 0 # iteration in terms of refinement of frames for training
TRAINING_SET_INDEX = 0
SUBDIR_STR = 'data_augm_00_baseline'

length_for_normalisation_in_px = 18 # pixels,  for Horses: median_eye2nose_length_px
pck_thresh_fraction = 0.2
pck_thresh_in_pixels = pck_thresh_fraction*length_for_normalisation_in_px #------

##########################################
# Compute paths to subdir
cfg = auxiliaryfunctions.read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
training_datasets_path = os.path.join(project_path, 
                                      SUBDIR_STR, 
                                      "training-datasets")
unaugmented_training_dataset_path = os.path.join(SUBDIR_STR,
                                                 auxiliaryfunctions.GetTrainingSetFolder(cfg))
                                               

##########################################
# Read human labelled data for this project
iteration_folder = os.path.join(training_datasets_path, 
                                'iteration-' + str(TRAIN_ITERATION))
dataset_top_folder = os.path.join(iteration_folder, os.listdir(iteration_folder)[0])
files_in_dataset_top_folder = os.listdir(dataset_top_folder)

h5_files_in_dataset_top_folder = \
        [el for el in files_in_dataset_top_folder if  el.endswith('h5')]
if len(h5_files_in_dataset_top_folder) > 1:
        print('More than one h5 file found at {}, selecting first one: {}'.format(dataset_top_folder,
                                                                                     h5_files_in_dataset_top_folder[0]))
human_labels_filepath = os.path.join(dataset_top_folder,
                                     h5_files_in_dataset_top_folder[0])

# ground truth labels
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

##############################################################
# Mean and sigma per bodypart
dict_px_error_bodypart_per_shuffle = dict()
dict_px_error_total_per_shuffle = dict()
for shuffle in list_shuffle_numbers:
        df_results  = dict_df_results_per_shuffle[shuffle]
        df_summary_per_bodypart = df_results.describe()
        dict_px_error_bodypart_per_shuffle.update({shuffle:df_summary_per_bodypart})
        print('----------------------')
        print('Shuffle {}, pixel error per bodypart:'.format(shuffle))
        print(df_summary_per_bodypart)
        

        # Mean and sigma across all bodyparts
        px_error_all_bodyparts_and_test_samples = np.nanmean(df_results.drop(labels='likelihood',axis=1,level=1)) # matches result for evaluate fn
        dict_px_error_total_per_shuffle.update({shuffle:px_error_all_bodyparts_and_test_samples})
        print('----------------------')
        print('Shuffle {}, pixel error all bodyparts and test samples:'.format(shuffle))
        print(px_error_all_bodyparts_and_test_samples)
        print('----------------------')
######################################################
# %%
# Plot px error per bodypart and shuffle
dict_data_per_bprt = dict_px_error_bodypart_per_shuffle
dict_data_total = dict_px_error_total_per_shuffle
dict_df = dict()
colors =['tab:blue','tab:orange','tab:green']
for i,shuffle in enumerate(list_shuffle_numbers):
        df = dict_data_per_bprt[shuffle]
        df = df.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1) 
        dict_df.update({i:df.T['mean']})

# prepare dataframe for plot      
df_for_plot = pd.concat([pd.DataFrame.from_dict(dict_df)], 
                        keys=['shuffle'], axis=1)
df_for_plot = df_for_plot.stack().reset_index()
df_for_plot.rename(columns={'level_1': 'shuffle',
                            'shuffle': 'normalised test RMSE'}, 
                   inplace=True)
# normalise error!
df_for_plot['normalised test RMSE'] = df_for_plot['normalised test RMSE']/length_for_normalisation_in_px                  
# plt figure
plt.figure(figsize=(12,10))
sns.swarmplot(data=df_for_plot,
                x='bodyparts',
                y='normalised test RMSE',
                size = 10, 
                hue='shuffle',
                palette = dict(zip(list_shuffle_numbers,colors))) #  x='bodyparts', y='mean',


# plot mean per shuffle
for i,shuffle in enumerate(list_shuffle_numbers):
      plt.axhline(y=dict_data_total[shuffle]/length_for_normalisation_in_px                  ,
                  linestyle='--',
                  color=colors[i])                 
plt.xticks(rotation=45,fontsize = 10, )
plt.ylim((0.05,0.3))
plt.grid()
plt.show()

###########################################################
# %%
# Compute PCK per shuffle
dict_pck_bodypart_per_shuffle = dict()
dict_pck_total_per_shuffle = dict()
for shuffle in list_shuffle_numbers:

        df_results  = dict_df_results_per_shuffle[shuffle]
        df_distance_per_bdpt = df_results.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1) # drop llk and distance level

        list_bdpts = list(df_distance_per_bdpt.columns.values)
        dict_pck_bodypart_per_shuffle.update({shuffle:{}})

        n_kpts_below_th_total = 0
        n_kpts_visible_total = 0
        # PCK per bodypart
        for bdpt in list_bdpts:
                # n of kpts below th for one bodypart
                n_kpts_below_th_one_bdpt = df_distance_per_bdpt[bdpt][df_distance_per_bdpt[bdpt] < pck_thresh_in_pixels].count()
                n_kpts_visible_one_bdpt = df_distance_per_bdpt[bdpt].count()
                dict_pck_bodypart_per_shuffle[shuffle].update({bdpt: n_kpts_below_th_one_bdpt/n_kpts_visible_one_bdpt})

                n_kpts_below_th_total += n_kpts_below_th_one_bdpt
                n_kpts_visible_total += n_kpts_visible_one_bdpt
        
        # PCK overall
        dict_pck_total_per_shuffle.update({shuffle: n_kpts_below_th_total/n_kpts_visible_total})#sum(dict_pck_bodypart_per_shuffle[shuffle].values())})



######################################################
# %%
# Plot PCK per bodypart and shuffle
dict_data_per_bprt = dict_pck_bodypart_per_shuffle
dict_data_total = dict_pck_total_per_shuffle
# dict_df = dict()
# colors =['tab:blue','tab:orange','tab:green']
# for i,shuffle in enumerate(list_shuffle_numbers):
#         df = dict_data_per_bprt[shuffle]
#         df = df.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1) 
#         dict_df.update({i:df.T['mean']})

# prepare dataframe for plot      
df_for_plot = pd.concat([pd.DataFrame.from_dict(dict_pck_bodypart_per_shuffle)], 
                        keys=['shuffle'], axis=1)
df_for_plot = df_for_plot.stack().reset_index()
df_for_plot.rename(columns={'level_0': 'bodyparts',
                            'level_1': 'shuffle',
                            'shuffle': 'PCK'+str(int(pck_thresh_fraction*100))}, 
                   inplace=True)
# plt figure
plt.figure(figsize=(12,10))
sns.swarmplot(data=df_for_plot,
                x='bodyparts',
                y='PCK'+str(int(pck_thresh_fraction*100)),
                size = 10, 
                hue='shuffle',
                palette = dict(zip(list_shuffle_numbers,colors))) #  x='bodyparts', y='mean',

# plt.axhline(y=0.9, color='r', linestyle=':')
# plot mean per shuffle
for i,shuffle in enumerate(list_shuffle_numbers):
      plt.axhline(y=dict_pck_total_per_shuffle[shuffle]                 ,
                  linestyle='--',
                  color=colors[i])                 
plt.xticks(rotation=45,fontsize = 10, )
plt.ylim((0.7,1))
plt.legend(loc='lower right')
plt.grid()
plt.show()

# %%
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

######################################################
# Save results as pickle file?
# dict_df_results_per_shuffle
# dict_pck
# dict_px_error_by_bodypart = dict()
# dict_px_error_total = dict()
