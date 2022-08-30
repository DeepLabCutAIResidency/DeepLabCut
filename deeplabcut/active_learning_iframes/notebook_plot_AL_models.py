###############################################################
# %% 
import sys
import os
from turtle import fillcolor
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

###############################################################
# %% Inputs

parent_dir_path = '/home/sofia/datasets' #'/home/sofia/datasets/Horse10_AL_unif_OH'
parent_dir_path_list = ['Horse10_AL_unif_small_OH',#'Horse10_AL_unif_OH',
                        'Horse10_AL_uncert_kmeans_OH','Horse10_AL_uncert_kmeans_rev_OH']
model_prefix_list = [x[:-3] for x in parent_dir_path_list] #'Horse10_AL_infl' #'Horse10_AL_unif000'

parent_dir_path_list = [os.path.join(parent_dir_path,x) for x in parent_dir_path_list]

TRAIN_ITERATION = 0
NUM_SHUFFLES = 3


###############################################################
# %%

plt.figure(figsize=(7.5,5.5))
list_md_fractions = np.arange(0.,0.3,0.05) #np.arange(0,1.25,0.25)
dict_md_colors = {k:v for k,v in zip(model_prefix_list,
                                    ['tab:blue','tab:green','tab:orange'])}
list_shuffle_numbers = range(1,NUM_SHUFFLES+1)

path_to_pickle_px_error_unif = '/home/sofia/datasets/Horse10_OH_outputs/Horse10_AL_unif_px_error.pkl'

for (model_prefix,parent_dir) in zip(model_prefix_list,
                                     parent_dir_path_list):
    # get list of AL models within this strategy                                 
    list_models_dirs = [x for x in os.listdir(parent_dir) 
                            if x.startswith(model_prefix) and not x.endswith('pkl')]  
    list_models_dirs.sort()
    # get normalised pixel error for the models in this strategy
    path_to_pickle_px_error = os.path.join('/home/sofia/datasets/Horse10_OH_outputs', #parent_dir,
                                           model_prefix+'_px_error.pkl')
    with open(path_to_pickle_px_error,'rb') as file:
        [_, dict_norm_px_error_total_per_model,_,_] = pickle.load(file)

    #---------------------------------------- 
    # if 'unif' in model_prefix:
    #     with open(path_to_pickle_px_error_unif,'rb') as file:
    #         [_, dict_norm_px_error_total_per_model2,_,_] = pickle.load(file)
    #         dict_norm_px_error_total_per_model.update(dict_norm_px_error_total_per_model2)
    #----------------------------------------
    # plot
    if model_prefix == 'Horse10_AL_unif':
        for sh in list_shuffle_numbers:
            plt.plot(list_md_fractions,
                     [dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                     color=dict_md_colors[model_prefix],
                     marker='o',
                     label=model_prefix)
    else:                                   
        for sh in list_shuffle_numbers:
            plt.plot(list_md_fractions,
                     [np.nan]+[dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                     color=dict_md_colors[model_prefix],
                     marker='o',# markerfacecolor=dict_sh_colors[sh], markeredgecolor=dict_sh_colors[sh],
                     label=model_prefix)
    plt.ylim([0,2])
    plt.xticks(list_md_fractions,
            [str(int(x*100)) for x in list_md_fractions])
    plt.xlabel('active learning frames (%)')
    plt.ylabel('normalised RMSE')

plt.legend()
plt.show()
###############################################################
# %% Plot normalised RMSE for each AL model

plt.figure(figsize=(7.5,5.5))

list_md_fractions = np.arange(0.,0.3,0.05) #np.arange(0,1.25,0.25)
dict_sh_colors = {k:v for k,v in zip(list_shuffle_numbers,
                                    ['tab:gray','tab:purple','tab:red'])}
list_shuffle_numbers = range(1,NUM_SHUFFLES+1)
for (model_prefix,parent_dir) in zip(model_prefix_list,
                                     parent_dir_path_list):
    # get list of AL models within this strategy                                 
    list_models_dirs = [x for x in os.listdir(parent_dir) if x.startswith(model_prefix) and not x.endswith('pkl')]  
    list_models_dirs.sort()
    # get normalised pixel error for the models in this strategy
    path_to_pickle_px_error = os.path.join('/home/sofia/datasets/Horse10_OH_outputs', #parent_dir,
                                           model_prefix+'_px_error.pkl')
    with open(path_to_pickle_px_error,'rb') as file:
        [_, dict_norm_px_error_total_per_model,_,_] = pickle.load(file)


    # plot
    if model_prefix == 'Horse10_AL_unif':
        for sh in list_shuffle_numbers:
            plt.plot(list_md_fractions,
                     [dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                     color=dict_sh_colors[sh],
                     marker='o',
                     linestyle='',
                     label='shuffle {}'.format(sh))
    else:                                   
        for sh in list_shuffle_numbers:
            plt.plot(list_md_fractions,
                     [np.nan]+[dict_norm_px_error_total_per_model[md][sh] for j,md in enumerate(list_models_dirs)],
                     color=dict_sh_colors[sh],
                     marker='o',
                     linestyle='',
                     label='shuffle {}'.format(sh))
    plt.ylim([0,2])
    plt.xticks(list_md_fractions,
            [str(int(x*100)) for x in list_md_fractions])
    plt.xlabel('active learning frames (%)')
    plt.ylabel('normalised RMSE')

plt.legend()
plt.show()
# %%
########################################################################
# %% Plot top3 and bottom 3 most influential/uncertain
import os
import pandas as pd
import pickle 
import matplotlib.pyplot as plt

reference_dir_path = '/home/sofia/datasets/Horse10_AL_uncert_OH' 
path_to_h5_file = os.path.join(reference_dir_path,  
                              'training-datasets/iteration-0/',
                              'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # do not use h5 in labeled-data!

path_to_pickle_w_AL_train_idcs_ranked_by_X = os.path.join(reference_dir_path,
                                                             'horses_AL_OH_train_uncert_ranked_idcs.pkl')
df_groundtruth = pd.read_hdf(path_to_h5_file)


with open(path_to_pickle_w_AL_train_idcs_ranked_by_X,'rb') as file:
    map_shuffle_id_to_AL_train_idcs_ranked = pickle.load(file)

for sh in [1,2,3]:
    list_top_3 = list(df_groundtruth.iloc[map_shuffle_id_to_AL_train_idcs_ranked[sh][:3],:].index)
    list_bottom_3 = list(df_groundtruth.iloc[map_shuffle_id_to_AL_train_idcs_ranked[sh][-3:],:].index)
    print('----------------')
    print(f'Shuffle {sh} top 3:')
    for j,el in enumerate(list_top_3):
            im=plt.imread(os.path.join(reference_dir_path,el))
            plt.imshow(im)
            plt.title('Shuffle {}, {} from top ({})'.format(sh,j+1,el))
            plt.show()
    print('----------------')
    print(f'Shuffle {sh} bottom 3:')
    for j,el in enumerate(list_bottom_3):
            im=plt.imread(os.path.join(reference_dir_path,el))
            plt.imshow(im)
            plt.title('Shuffle {}, {} from bottom ({})'.format(sh,j+1,el))
            plt.show()
    # df_groundtruth.index(map_shuffle_id_to_AL_train_idcs_ranked[sh][-3:])

######################################################
# %% Plot MPE vs RMSE

path_to_pickle_w_mpe_per_shuffle = '/home/sofia/datasets/Horse10_AL_unif_OH/horses_AL_OH_mpe_per_shuffle_snapshotID_neg1.pkl'
# f'horses_AL_OH_mpe_per_shuffle_snapshotID_{snapshot_idx_str}.pkl'
path_to_pickle_w_rmse_AL_unif_models = '/home/sofia/datasets/Horse10_AL_unif_OH/Horse10_AL_unif_px_error.pkl'

## load MPE results on OOD test set per shuffle
with open(path_to_pickle_w_mpe_per_shuffle,'rb') as file:
    [dict_df_test_only_w_uncert_per_shuffle] = pickle.load(file)
list_shuffles = list(dict_df_test_only_w_uncert_per_shuffle.keys())

## load rmse in unif models
with open(path_to_pickle_w_rmse_AL_unif_models,'rb') as file:
  [dict_px_error_total_per_AL_unif_model,
    dict_norm_px_error_total_per_AL_unif_model,
    dict_df_results_per_AL_unif_model,
    dict_px_error_bodypart_per_AL_unif_model] = pickle.load(file)
###################################
# %%
## plot distance per bprt and MPE per bdpimagert
dict_df_error_AL_unif000_per_shuffle = dict_df_results_per_AL_unif_model['Horse10_AL_unif000']
dict_sh_colors = {k:v for k,v in zip(list_shuffles,
                                    ['tab:orange','tab:purple','tab:green'])}
plt.figure(figsize=(7,7))                                    
for sh in list_shuffles:
    df_mpe_test_OOD_samples = dict_df_test_only_w_uncert_per_shuffle[sh]
    df_error_test_OOD_samples = dict_df_error_AL_unif000_per_shuffle[sh]
    if not (df_mpe_test_OOD_samples.index == df_error_test_OOD_samples.index).all():
        print('Mismatch between samples')
        sys.exit()
    # get list of bodyparts
    list_bdprts = list(set([x[1] for x in df_mpe_test_OOD_samples.columns]))
    list_bdprts = [x for x in list_bdprts if x != '']
    # list_bdprts_rep_from_df = [x[1] for x in df_mpe_test_OOD_samples.columns] # from original df columns, with repetitions
    # list_bdprt_loc_in_orig_df = [list_bdprts_rep_from_df.index(y) for y in list_bdprts]
    # list_bdprts_srted_as_df_cols = [x for x,_ in sorted(zip(list_bdprts,list_bdprt_loc_in_orig_df),
    #                                                     key=lambda pair:pair[1])]

    # plot   
    for bdprt in list_bdprts: 
        plt.scatter(df_mpe_test_OOD_samples.loc[:,('Byron',bdprt,'MPE_bdprt')],
                    df_error_test_OOD_samples.loc[:,(bdprt,'distance')], #distance_norm
                    10,
                    dict_sh_colors[sh],
                    label=sh)  
# plt.ylim([0, 1])
plt.xlabel('MPE per predicted kpt')
plt.ylabel('normalised distance kpt to groundtruth')
plt.legend([str(x) for x in list_shuffles])  
plt.show()
###################################
# %%
## plot distance per bprt and MPE per bdprt
dict_df_error_AL_unif000_per_shuffle = dict_df_results_per_AL_unif_model['Horse10_AL_unif000']
dict_sh_colors = {k:v for k,v in zip(list_shuffles,
                                    ['tab:orange','tab:purple','tab:green'])}
# plt.figure(figsize=(7,7))                                    
for sh in list_shuffles:
    plt.figure(figsize=(7,7))  
    # get dataframes
    df_mpe_test_OOD_samples = dict_df_test_only_w_uncert_per_shuffle[sh]
    df_error_test_OOD_samples = dict_df_error_AL_unif000_per_shuffle[sh]
    if not (df_mpe_test_OOD_samples.index == df_error_test_OOD_samples.index).all():
        print('Mismatch between samples')
        sys.exit()
    # group by images
    list_bdprts = list(set([x[1] for x in df_mpe_test_OOD_samples.columns]))
    list_bdprts = [x for x in list_bdprts if x != '']
    list_cols_for_mean = [('Byron',bp,'MPE_bdprt') for bp in list_bdprts]
    df_mpe_test_OOD_samples['mean_MPE_per_img'] = df_mpe_test_OOD_samples[list_cols_for_mean].mean(axis=1)
    df_mpe_test_OOD_samples['max_MPE_per_img'] = df_mpe_test_OOD_samples[list_cols_for_mean].max(axis=1)
    # check
    print(np.all(np.array(df_mpe_test_OOD_samples['mean_MPE_per_img'] == df_mpe_test_OOD_samples['mean_MPE'])))
    
    list_cols_for_mean_error = [(bp,'distance_norm') for bp in list_bdprts]
    df_error_test_OOD_samples['mean_norm_error'] = df_error_test_OOD_samples[list_cols_for_mean_error].mean(axis=1)
    df_error_test_OOD_samples['max_norm_error'] = df_error_test_OOD_samples[list_cols_for_mean_error].max(axis=1)

    # plot 
    plt.hlines(0.5, 1.50,1.60)
    plt.scatter(df_mpe_test_OOD_samples['max_MPE_per_img'],
                df_error_test_OOD_samples['max_norm_error'], #distance_norm
                10,
                dict_sh_colors[sh],
                label=sh,
                alpha=0.1)  
# plt.ylim([0, 1])
plt.xlabel('max MPE per predicted img')
plt.ylabel('max norm error per img')
plt.legend([str(x) for x in list_shuffles])  
plt.show()

###################################
# %% as a 2D histogram
dict_df_error_AL_unif000_per_shuffle = dict_df_results_per_AL_unif_model['Horse10_AL_unif000']
dict_sh_colors = {k:v for k,v in zip(list_shuffles,
                                    ['tab:orange','tab:purple','tab:green'])}
fig = plt.figure(figsize=(7,7))                                 
for sh in list_shuffles:
    df_mpe_test_OOD_samples = dict_df_test_only_w_uncert_per_shuffle[sh]
    df_error_test_OOD_samples = dict_df_error_AL_unif000_per_shuffle[sh]
    if not (df_mpe_test_OOD_samples.index == df_error_test_OOD_samples.index).all():
        print('Mismatch between samples')
        sys.exit()
    # get list of bodyparts
    list_bdprts = list(set([x[1] for x in df_mpe_test_OOD_samples.columns]))
    list_bdprts = [x for x in list_bdprts if x != '']

    # plot 
    list_mpe_per_kpts_and_sample = []
    list_dist_norm_per_kpt_and_sample = []    

    for bp_i,bd_str in enumerate(list_bdprts): 
        list_mpe_per_kpts_and_sample.append(\
            np.array(df_mpe_test_OOD_samples.loc[:,('Byron',bdprt,'MPE_bdprt')]))
        list_dist_norm_per_kpt_and_sample.append(\
            np.array(df_error_test_OOD_samples.loc[:,(bdprt,'distance')])) # distance_norm
    
    mpe_array = np.stack(list_mpe_per_kpts_and_sample,axis=1)
    dist_norm_array = np.stack(list_dist_norm_per_kpt_and_sample,axis=1)

    # unflatten all bodyparts together and remove nans
    mpe_array_2d_hist = mpe_array.flatten()
    dist_norm_array_2d_hist = dist_norm_array.flatten()

    slc_not_nan_in_both = np.logical_and(~np.isnan(mpe_array_2d_hist),
                                         ~np.isnan(dist_norm_array_2d_hist))
    mpe_array_2d_hist = mpe_array_2d_hist[slc_not_nan_in_both]
    dist_norm_array_2d_hist = dist_norm_array_2d_hist[slc_not_nan_in_both]

    # plt.scatter(mpe_array_2d_hist,
    #            dist_norm_array_2d_hist,50,'k')
    plt.hist2d(mpe_array_2d_hist,
               dist_norm_array_2d_hist,
               bins = [10,5],
               zorder=-1)  
    plt.xlabel('MPE per predicted kpt')
    plt.ylabel('normalised distance kpt to groundtruth')
    plt.title(f'Shuffle {sh}')
    # plt.legend([str(x) for x in list_shuffles])   
    plt.colorbar()
    plt.show()
