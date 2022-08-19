"""
Script to create base dataset for active learning study.
We divide the dataset in 2 parts: a train set, made up of 10 horses, and a test set, made up of 20 horses. 
The complete test set will be OOD.

For each shuffle: we train on one horse from the train set + a subset selected from the remaining horses from the train set.

The script saves the idcs per shuffle to pickle files.

- Before running this script, prepare directory structure:
    - For example for the active learning with uniform samplingd ownload horse10.tar.gz and 
      extract in 'Horse10_AL_unif_OH': (use --strip-components 1: OH= oneHorse, refers to base train set being one horse only)
        mkdir Horse10_AL_unif_OH
        tar -xvzf horse10.tar.gz -C /home/sofia/datasets/Horse10_AL_unif_OH --strip-components 1

"""
# %%
import os
import sys
import pickle
import pandas as pd
import re

from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import compute_idcs_from_list_dirs
from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import compute_dicts_w_nfiles_in_train_test_sets_per_shuffle
from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import convert_dir_list_to_dict_w_nfiles

#%%
###################################################
if __name__ == "__main__":
    ####################################
    # %%
    ### Input data
    parent_dir_path = '/home/sofia/datasets/Horse10_AL_unif_OH/labeled-data'
    h5_file_path = '/home/sofia/datasets/Horse10_AL_unif_OH/training-datasets/iteration-0/'+\
                    'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
    output_dirs_pickle_path = '/home/sofia/datasets/Horse10_AL_unif_OH/horses_AL_OH_train_test_dirs_split.pkl'
    output_idcs_pickle_path = '/home/sofia/datasets/Horse10_AL_unif_OH/horses_AL_OH_train_test_idcs_split.pkl'

    FILES_EXT = '.png' # extension of files in subdirectories

    
    ###############################################################
    # %%
    ### Define dict with all train directories per shuffle 
    map_shuffle_id_to_all_train_dirs = dict()
    # All train directories for shuffle 1
    map_shuffle_id_to_all_train_dirs[1] = ['ChestnutHorseLight',
                                            'GreyHorseLightandShadow',
                                            'GreyHorseNoShadowBadLight',
                                            'Sample12',
                                            'Sample13',
                                            'Sample17',
                                            'Sample5',
                                            'Sample7',
                                            'Sample8',
                                            'Twohorsesinvideoonemoving']
    # All train directories for shuffle 2
    map_shuffle_id_to_all_train_dirs[2] = ['Brownhorselight',
                                            'Brownhorseoutofshadow',
                                            'ChestnutHorseLight',
                                            'Chestnuthorseongrass',
                                            'GreyHorseLightandShadow',
                                            'Sample11',
                                            'Sample13',
                                            'Sample18',
                                            'Sample8',
                                            'Twohorsesinvideoonemoving']

    # All train directories for shuffle 3
    map_shuffle_id_to_all_train_dirs[3] = ['BrownHorseinShadow',
                                            'Brownhorseoutofshadow',
                                            'Chestnuthorseongrass',
                                            'GreyHorseNoShadowBadLight',
                                            'Sample12',
                                            'Sample13',
                                            'Sample19',
                                            'Sample4',
                                            'TwoHorsesinvideobothmoving',
                                            'Twohorsesinvideoonemoving']

    #################################################################
    # %%
    ## Define base train set per shuffle   --ATT! values are lists (of one element if only one horse is used for b ase training)
    # if one horse: selected (manually) as the videos with largest number of frames, over all train directories across all shuffles 
    map_shuffle_id_to_base_train_dirs = {1:['Sample8'],
                                         2:['GreyHorseLightandShadow'],
                                         3:['Chestnuthorseongrass']}         
    # check
    for sh in map_shuffle_id_to_all_train_dirs.keys():
        if not all([x in map_shuffle_id_to_all_train_dirs[sh] for x in map_shuffle_id_to_base_train_dirs[sh]]):
            print('ERROR: the base train directories for shuffle {} are not part of the full train set'. format(sh))   
        else:
            n_dirs = len(map_shuffle_id_to_base_train_dirs[sh])
            print('Base train dirs ({} in total) included in full train set for shuffle {}'.format(n_dirs,sh))    
                                                                     
    #################################################################
    # %%
    # Compute active learning train set per shuffle
    # (made up of the remaining horses in full train set)
    map_shuffle_id_to_AL_train_dirs = dict()
    for sh in map_shuffle_id_to_all_train_dirs.keys():
        # AL train data: train data not in base train set
        map_shuffle_id_to_AL_train_dirs[sh] = [el for el in map_shuffle_id_to_all_train_dirs[sh]
                                                  if el not in map_shuffle_id_to_base_train_dirs[sh]]

    ##################################################################
    # %%
    # Compute OOD test set per shuffle

    # Get all (train+test) directories and their number of files 
    list_all_dirs = [el for el in os.listdir(parent_dir_path) if os.path.isdir(os.path.join(parent_dir_path,el))] # only parent dirs (30)
    list_all_dirs.sort() # sort just for convenience
    if len(list_all_dirs) != 30:
        print('ERROR: not exactly 30 horses in dataset')
        sys.exit(1)
    # convert to dict with nfiles per dir
    map_horses_dirs_to_nfiles = convert_dir_list_to_dict_w_nfiles(list_all_dirs,
                                                                  parent_dir_path,
                                                                  FILES_EXT)
    # Compute OOD test set for each shuffle
    map_shuffle_id_to_OOD_test_dirs = dict()
    for sh in map_shuffle_id_to_all_train_dirs.keys():
        # test data: data not in full train set
        map_shuffle_id_to_OOD_test_dirs[sh] = [el for el in list_all_dirs
                                               if el not in map_shuffle_id_to_all_train_dirs[sh]]

    ##################################################################
    # %%
    # Compute dicts with number of files per dir, per shuffle
    n_shuffles = len(map_shuffle_id_to_all_train_dirs.keys())

    map_shuffle_id_to_base_train_dirs_w_nfiles = dict()
    map_shuffle_id_to_AL_train_dirs_w_nfiles = dict()
    map_shuffle_id_to_OOD_test_dirs_w_nfiles = dict()
    for map_out, map_in in zip([map_shuffle_id_to_base_train_dirs_w_nfiles,
                                 map_shuffle_id_to_AL_train_dirs_w_nfiles,
                                 map_shuffle_id_to_OOD_test_dirs_w_nfiles],
                                [map_shuffle_id_to_base_train_dirs,
                                 map_shuffle_id_to_AL_train_dirs,
                                 map_shuffle_id_to_OOD_test_dirs]):
        
        
        for sh in range(1,n_shuffles+1):
            map_out[sh] = convert_dir_list_to_dict_w_nfiles(map_in[sh], #  list of subdirs
                                                            parent_dir_path,
                                                            FILES_EXT)
                                                                            

    #########################################################
    # %%
    ## Compute dicts with idcs per shuffle, for each subset of the dataset
    map_shuffle_id_to_base_train_idcs = dict()
    map_shuffle_id_to_AL_train_idcs = dict()
    map_shuffle_id_to_OOD_test_idcs = dict()

    df = pd.read_hdf(h5_file_path)

    for map_out,map_in in zip([map_shuffle_id_to_base_train_idcs,
                                map_shuffle_id_to_AL_train_idcs,
                                map_shuffle_id_to_OOD_test_idcs],
                              [map_shuffle_id_to_base_train_dirs_w_nfiles,
                                map_shuffle_id_to_AL_train_dirs_w_nfiles,
                                map_shuffle_id_to_OOD_test_dirs_w_nfiles]):

        for sh in range(1,n_shuffles+1):                        
            map_out[sh] = compute_idcs_from_list_dirs(map_in[sh],
                                                      df)   

            # check
            if len(map_out[sh]) != sum(map_in[sh].values()):
                print('ERROR: number of idcs extracted and number of files in subset of directories do not match for shuffle {}'.format(sh))
    
    #----------------
    # check idcs in dataframe corresp to correct dirs
    # for map_idcs,map_dirs in zip([map_shuffle_id_to_base_train_idcs,
    #                             map_shuffle_id_to_AL_train_idcs,
    #                             map_shuffle_id_to_OOD_test_idcs],
    #                           [map_shuffle_id_to_base_train_dirs_w_nfiles,
    #                             map_shuffle_id_to_AL_train_dirs_w_nfiles,
    #                             map_shuffle_id_to_OOD_test_dirs_w_nfiles]):
    #     for sh in range(1,n_shuffles+1): 
    #         l_dirs_from_df = list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  
    #                                     for el in map_idcs[sh]]))
    #         if not (l_dirs_from_df.sort() == list(map_dirs[sh].keys()).sort()):
    #             print('ERROR: mismatch between directories from dataframe and directories from os')
    #         else:
    #             print('Dirs from df indices match os dirs')
    ########################################################
    # %%
    # Save results as pickle
    # dirs per shuffle
    with open(output_dirs_pickle_path,'wb') as file:
        pickle.dump([map_shuffle_id_to_base_train_dirs_w_nfiles,
                     map_shuffle_id_to_AL_train_dirs_w_nfiles,
                     map_shuffle_id_to_OOD_test_dirs_w_nfiles], file)

    # idcs per shuffle
    with open(output_idcs_pickle_path,'wb') as file:
        pickle.dump([map_shuffle_id_to_base_train_idcs,
                     map_shuffle_id_to_AL_train_idcs,
                     map_shuffle_id_to_OOD_test_idcs], file)
# %%
