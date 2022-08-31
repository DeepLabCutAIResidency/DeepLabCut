"""
TODO
- add PCK
"""

#########################################
# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import re
import math

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

import os, sys
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
import re 
import argparse
import yaml
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # to supress future warnings...

#########################################
# %%
# Input params
AL_strategy_str = 'AL_unif' # uniform sampling: AL_unif; AL_uncert; AL_infl 'unif' # 'unif', 'uncert'
parent_dir_path =  f'/home/sofia/datasets/Horse10_{AL_strategy_str}_OH' #'/home/sofia/datasets/Horse10_AL_{}_OH'.format(AL_strategy_str) #'/home/sofia/datasets/Horse10_AL_unif_OH'
model_prefix = f'Horse10_{AL_strategy_str}' #'Horse10_AL_unif000'
pickle_output_path = os.path.join('/home/sofia/datasets/Horse10_OH_outputs',
                                  model_prefix+'_px_error.pkl')

# length_for_normalisation_in_px = 18 # pixels,  for Horses: median_eye2nose_length_px
# pck_thresh_fraction = 0.3
# pck_thresh_in_pixels = pck_thresh_fraction*length_for_normalisation_in_px #------

TRAIN_ITERATION = 0 # iteration in terms of frames extraction; default is 0, but in stinkbug is 1. can this be extracted?


## Set 'allow growth' before eval (allow growth bug)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


##############################################
# %%
## Compute normalised error for all shuffles per model
dict_df_results_per_model = dict()
dict_px_error_bodypart_per_model = dict()
dict_px_error_total_per_model = dict()
dict_norm_px_error_total_per_model = dict()

list_models_dirs = [el for el in os.listdir(parent_dir_path) 
                       if el.startswith(model_prefix) and not el.endswith('pkl')]
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
        # dict_training_fraction_idx_per_shuffle = {}   
        for sh in list_shuffle_numbers:
                dict_training_fraction_per_shuffle[sh] =[float(re.search('_([0-9]*)shuffle{}.pickle'.format(sh), el).group(1))/100
                                                                for el in list_pickle_files_in_dataset_top_folder
                                                                if 'shuffle{}.pickle'.format(sh) in el][0]
                # dict_training_fraction_idx_per_shuffle[sh] = \
                #         list_train_fractions_from_config.index(dict_training_fraction_per_shuffle[sh])

                                                
        ##########################################
        ## Read human labelled data for this model (common to all shuffles)---and actually models too
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

        # ground truth labels for this model
        df_human = pd.read_hdf(human_labels_filepath)

        #############################################################
        ## Compute normalising factor per video
        df_human.loc[:,'sq_diff_x'] = (df_human.loc[:,('Byron', 'Eye','x')] - df_human.loc[:,('Byron', 'Nose','x')])**2
        df_human.loc[:,'sq_diff_y'] = (df_human.loc[:,('Byron', 'Eye','y')] - df_human.loc[:,('Byron', 'Nose','y')])**2
        df_human.loc[:,'eye2nose'] = np.sqrt(df_human.loc[:,'sq_diff_x'] + df_human.loc[:,'sq_diff_y'])
        
        df_human.loc[:,'video'] =  [x.split('/')[1] for x in df_human.index]
        # plt.plot(df_human.loc[:,'eye2nose'] )
        # plt.show()

        df_normalisation = df_human.groupby([('video','','')]).median()
        # float(df.loc['BrownHorseinShadow','eye2nose'])
        ##############################################################
        ## Compute dataframe with error per keypoint for each shuffle
        # Loop thru shuffles (different predictions per shuffle)
        config_path_one_model = os.path.join(parent_dir_path, 
                                             md,
                                             'config.yaml')
        dict_df_results_per_shuffle = dict()
        for sh in list_shuffle_numbers: #[list_shuffle_numbers[0]]: #range(1,NUM_SHUFFLES+1):

                ### Get predictions for the last snapshot for this shuffle
                path_to_eval_snapshot_parent_dir =\
                        auxiliaryfunctions.get_evaluation_folder(dict_training_fraction_per_shuffle[sh], #list_training_fractions[TRAINING_SET_INDEX],
                                                                 sh, 
                                                                 cfg, 
                                                                 modelprefix=md)  
                
                list_snapshots = [el for el in os.listdir(os.path.join(parent_dir_path, #project_path,
                                                                      str(path_to_eval_snapshot_parent_dir))) 
                                     if el.endswith(".h5")]

                list_snapshot_iter = [int(re.findall('[0-9]+.h5',el)[0].split('.')[0]) 
                                      for el in list_snapshots]
                idx_latest_snapshot = np.argmax(np.array(list_snapshot_iter))                                                     
                model_predictions_one_shuffle_filepath = \
                        os.path.join(parent_dir_path, #project_path,
                                        str(path_to_eval_snapshot_parent_dir),
                                        list_snapshots[idx_latest_snapshot])

                ### Read model predictions (OJO predictions are computed for the full dataset, not just test samples)
                df_model_one_shuffle = pd.read_hdf(model_predictions_one_shuffle_filepath)

                ### Get test indices for this shuffle (from pickle?)
                _, pckl_metadata = auxiliaryfunctions.GetDataandMetaDataFilenames(dataset_top_folder, #unaugmented_training_dataset_path, 
                                                                                dict_training_fraction_per_shuffle[sh], #list_training_fractions[TRAINING_SET_INDEX], 
                                                                                sh, 
                                                                                cfg)
                _,_, testIndices, _ = auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"], 
                                                                      pckl_metadata))

                # Get rows from test set only ----  these df may not be sorted in the same way right? 
                # check match
                if df_human.index.to_list() != ['/'.join(x) for x in df_model_one_shuffle.index]:
                        print('ERROR: dataframes with human labels and model predictions have different rows')
                        sys.exit()
                df_human_test_only = df_human.iloc[testIndices,:]  # test idcs form images in ascending order?
                df_model_test_only = df_model_one_shuffle.iloc[testIndices,:] 

                # Drop scorer level
                df_human_test_only = df_human_test_only.droplevel('scorer',axis=1)
                df_model_test_only = df_model_test_only.droplevel('scorer',axis=1)   

                ### Compute x and y error: deltas in x and y dir between human scorer and model prediction
                # make indices match if req
                df_model_test_only.index = ['/'.join(x) for x in df_model_test_only.index]
                df_diff_test_only = df_human_test_only - df_model_test_only.drop(labels='likelihood',axis=1,level=1)
                

                #### Compute distance btw model and human
                # - nrows = samples in test set
                # - ncols = bodyparts tracked
                # Drop llk for model predictions before computing distance
                # df_diff_test_only = df_diff_test_only.drop(labels='likelihood',axis=1,level=1)
                df_distance_test_only = df_diff_test_only.pow(2).sum(level='bodyparts',axis=1,skipna=False).pow(0.5)
                # warning: recommends to use 'df_diff_test_only.pow(2).groupby(level='bodyparts',axis=1).sum(axis=1,skipna=False)' instead,
                # but that makes NaNs into 0s!
                # add distance level
                df_distance_test_only.columns = pd.MultiIndex.from_product([df_distance_test_only.columns, 
                                                                           ['distance']])

                #----------------------------------
                ### Add normalised error?
                list_videos_test_only = [x.split('/')[1] for x in df_distance_test_only.index]
                for bdprt in [x[0] for x in df_distance_test_only.columns]: # loop thru cols (aka bdprts)
                        df_distance_test_only.loc[:,(bdprt,'distance_norm')] = \
                        [d / float(df_normalisation.loc[video_str,'eye2nose']) 
                         for (d,video_str) in zip(df_distance_test_only.loc[:,(bdprt,'distance')],
                                                  list_videos_test_only)]
                #----------------------------------

                ## Combine w Likelihood
                df_llk_test_only = df_model_test_only.drop(labels=['x','y'],axis=1,level=1)
                df_results = pd.concat([df_distance_test_only,df_llk_test_only],axis=1).sort_index(level=0,axis=1)

                dict_df_results_per_shuffle.update({sh:df_results})

        ##############################################################
        # Mean and sigma per bodypart
        print('----------------------')
        print('Model {}'.format(md))
        print('----------------------')
        dict_px_error_bodypart_per_shuffle = dict()
        dict_px_error_total_per_shuffle = dict()
        dict_norm_px_error_total_per_shuffle = dict()
        for shuffle in list_shuffle_numbers:
                df_results  = dict_df_results_per_shuffle[shuffle]
                df_summary_per_bodypart = df_results.describe()
                dict_px_error_bodypart_per_shuffle.update({shuffle:df_summary_per_bodypart})
                print('----------------------')
                print('Shuffle {}, pixel error per bodypart:'.format(shuffle))
                print(df_summary_per_bodypart)
                

                # Mean and sigma across all bodyparts
                px_error_all_bodyparts_and_test_samples = np.nanmean(df_results.drop(labels=['distance_norm','likelihood'],axis=1,level=1)) # matches result for evaluate fn
                dict_px_error_total_per_shuffle.update({shuffle:px_error_all_bodyparts_and_test_samples})

                norm_px_error_all_bodyparts_and_test_samples = np.nanmean(df_results.drop(labels=['distance','likelihood'],axis=1,level=1)) # matches result for evaluate fn
                dict_norm_px_error_total_per_shuffle.update({shuffle:norm_px_error_all_bodyparts_and_test_samples})
                print('----------------------')
                print('Shuffle {}, (norm) pixel error all bodyparts and test samples:'.format(shuffle))
                print('({:.2f}) {:.2f}'.format(norm_px_error_all_bodyparts_and_test_samples,
                                               px_error_all_bodyparts_and_test_samples))
                print('----------------------')
        ######################################################
        # Add results to model dict
        dict_px_error_total_per_model[md] = dict_px_error_total_per_shuffle
        dict_norm_px_error_total_per_model[md] = dict_norm_px_error_total_per_shuffle
        dict_df_results_per_model[md] = dict_df_results_per_shuffle
        dict_px_error_bodypart_per_model[md] = dict_px_error_bodypart_per_shuffle
        
#############################################
# %%
with open(pickle_output_path,'wb') as file:
    pickle.dump([dict_px_error_total_per_model,
                 dict_norm_px_error_total_per_model,
                 dict_df_results_per_model,
                 dict_px_error_bodypart_per_model], file)

#############################################
# %% to load other data to plot:

# AL_strategy_str = 'AL_unif_small' # uniform sampling: AL_unif; AL_uncert; AL_infl 'unif' # 'unif', 'uncert'
# parent_dir_path =  f'/home/sofia/datasets/Horse10_{AL_strategy_str}_OH' #'/home/sofia/datasets/Horse10_AL_{}_OH'.format(AL_strategy_str) #'/home/sofia/datasets/Horse10_AL_unif_OH'
# model_prefix = f'Horse10_{AL_strategy_str}' #'Horse10_AL_unif000'

# list_models_dirs = [el for el in os.listdir(parent_dir_path) 
#                        if el.startswith(model_prefix) and not el.endswith('pkl')]
# list_models_dirs.sort()

# list_shuffle_numbers = [1,2,3]

# pickle_input_path = f'/home/sofia/datasets/Horse10_OH_outputs/Horse10_{AL_strategy_str}_px_error.pkl'
# with open(pickle_input_path,'rb') as file:
#     [dict_px_error_total_per_model,
#       dict_norm_px_error_total_per_model,
#        dict_df_results_per_model,
#        dict_px_error_bodypart_per_model] = pickle.load(file)
##########################################
# %% Plot results

                       
plt.figure(figsize=(7.5,5.5))
col = 'tab:blue'
list_md_fractions = np.arange(0.,0.3,0.05) #np.arange(0,1.25,0.25)
if model_prefix == 'Horse10_AL_unif':
        for sh in list_shuffle_numbers:
                plt.plot(list_md_fractions,
                        [dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                        color=col,
                        marker='o',
                        label=model_prefix)
else:                                   
        for sh in list_shuffle_numbers:
                plt.plot(list_md_fractions,
                        [np.nan]+[dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                        color=col,
                        marker='o',
                        label=model_prefix)
plt.legend([model_prefix])
plt.ylim([0,2])
plt.hlines(0.25,0,0.25,'r', linestyle='--')
plt.xticks(list_md_fractions,
        [str(int(x*100)) for x in list_md_fractions])
plt.xlabel('active learning frames (%)')
plt.ylabel('normalised RMSE')
plt.show()
# list_md_fractions = np.arange(0,1.25,0.25)
# for sh in list_shuffle_numbers:
#         plt.plot(list_md_fractions,
#                  [np.nan]+[dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
#                  color='tab:orange',
#                  marker='o',
#                  label=model_prefix)
# plt.legend(['infl'])
# plt.ylim([0,2])
# plt.xticks(list_md_fractions,
#           [str(int(x*100)) for x in list_md_fractions])
# plt.xlabel('active learning frames (%)')
# plt.ylabel('normalised RMSE')
# plt.show()
# %%
# list_md_fractions = [0,0.25,0.5,0.75,1.0]
# color_per_sh_str = ['tab:blue','tab:orange','tab:green']
# for j,md in enumerate(list_models_dirs):
#         for sh in list_shuffle_numbers:
#                 # norm_px_error_total_one_shuffle = dict_norm_px_error_total_per_model[md][sh]
#                 plt.scatter(list_md_fractions[j],
#                             [dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
#                             10,
#                             color_per_sh_str[sh-1])
# plt.show()



# %%
