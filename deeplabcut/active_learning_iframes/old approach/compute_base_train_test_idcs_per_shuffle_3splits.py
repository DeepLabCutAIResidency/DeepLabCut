"""
Script to create base dataset for active learning study.
It uses the initial approach in which we divided the dataset in 3 parts, of approx 10 horses each.
For each shuffle we have:
- 10 horses in the train set. These correspond to the training horses used in the OOD paper (but here we use all samples)
- 20 horses in the test set. The test set is divided in two parts, such that the difference in total number of frames between 
    both parts is min. One part is used for active learning and the other is used for evaluating performance.

It receives as input the list of videos to use for training for each shuffle.
It splits the remaining set of videos in two parts, such that the difference in total number of frames between them is minimum
(min sum partition problem). One of this parts is the Active Learning test set, the set of frames from which we will sample frames
in each active learning iteration. The other part is the Out of domain test set, which is made up of frames that are never seen
during training and we will use to compare the models' performance.

The script saves the idcs per shuffle to pickle files.

The solution to the min sum partition problem is adapted from:
https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/

"""
import os
import sys
import pickle
import pandas as pd
import re

from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import compute_idcs_from_list_dirs
from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import compute_dicts_w_nfiles_in_train_test_sets_per_shuffle
from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import split_test_set_dicts
# from deeplabcut.active_learning_iframes.custom_horse_dataset_utils import convert_dir_list_to_dict_w_nfiles


###################################################
# Driver
if __name__ == "__main__":
    ####################################
    ### Input data
    parent_dir_path = '/home/sofia/datasets/Horse10_AL/Horses-Byron-2019-05-08/labeled-data'
    h5_file_path = '/home/sofia/datasets/Horse10_AL/Horses-Byron-2019-05-08/training-datasets_/iteration-0/'+\
                    'UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
    output_dirs_pickle_path = '/home/sofia/datasets/Horse10_AL/Horses-Byron-2019-05-08/horses_AL_train_test_dirs_split.pkl'
    output_idcs_pickle_path = '/home/sofia/datasets/Horse10_AL/Horses-Byron-2019-05-08/horses_AL_train_test_idcs_split.pkl'

    FILES_EXT = '.png' # extension of files in subdirectories, with dot

    ### Compute dict with train directories per shuffle and number of files
    map_shuffle_id_to_train_dirs = dict()
    # Train directories for shuffle 1
    map_shuffle_id_to_train_dirs[1] = ['ChestnutHorseLight',
                                        'GreyHorseLightandShadow',
                                        'GreyHorseNoShadowBadLight',
                                        'Sample12',
                                        'Sample13',
                                        'Sample17',
                                        'Sample5',
                                        'Sample7',
                                        'Sample8',
                                        'Twohorsesinvideoonemoving']
    # Train directories for shuffle 2
    map_shuffle_id_to_train_dirs[2] = ['Brownhorselight',
                                        'Brownhorseoutofshadow',
                                        'ChestnutHorseLight',
                                        'Chestnuthorseongrass',
                                        'GreyHorseLightandShadow',
                                        'Sample11',
                                        'Sample13',
                                        'Sample18',
                                        'Sample8',
                                        'Twohorsesinvideoonemoving']

    # Train directories for shuffle 3
    map_shuffle_id_to_train_dirs[3] = ['BrownHorseinShadow',
                                        'Brownhorseoutofshadow',
                                        'Chestnuthorseongrass',
                                        'GreyHorseNoShadowBadLight',
                                        'Sample12',
                                        'Sample13',
                                        'Sample19',
                                        'Sample4',
                                        'TwoHorsesinvideobothmoving',
                                        'Twohorsesinvideoonemoving']
 
    ##################################################################
    # Compute dicts between dirs and number of files
    (map_horses_dirs_to_nfiles,
     map_shuffle_id_to_train_dirs_w_nfiles,
     map_shuffle_id_to_test_dirs_w_nfiles) = compute_dicts_w_nfiles_in_train_test_sets_per_shuffle(parent_dir_path,
                                                                                                    map_shuffle_id_to_train_dirs,
                                                                                                    FILES_EXT)
    list_all_dirs = map_horses_dirs_to_nfiles.keys()   
    n_shuffles = len(map_shuffle_id_to_train_dirs.keys())                                                                            

    ###############################################################
    ## Compute split in test set: divide test set into active learning set (AL) and OOD set, 
    # such that the number of samples per shuffle across both is balanced 
    (map_shuffle_id_to_test_AL_dirs_w_nfiles,
     map_shuffle_id_to_test_OOD_dirs_w_nfiles) = split_test_set_dicts(map_shuffle_id_to_test_dirs_w_nfiles)

    #########################################################
    ## Compute idcs per shuffle for each subset of the dataset
    map_shuffle_id_to_train_idcs = dict()
    map_shuffle_id_to_test_AL_idcs = dict()
    map_shuffle_id_to_test_OOD_idcs = dict()

    df = pd.read_hdf(h5_file_path)

    for map_out,map_in in zip([map_shuffle_id_to_train_idcs,
                                map_shuffle_id_to_test_AL_idcs,
                                map_shuffle_id_to_test_OOD_idcs],
                                [map_shuffle_id_to_train_dirs_w_nfiles, 
                                map_shuffle_id_to_test_AL_dirs_w_nfiles,
                                map_shuffle_id_to_test_OOD_dirs_w_nfiles]):

        for sh in range(1,n_shuffles+1):                        
            map_out[sh] = compute_idcs_from_list_dirs(map_in[sh],df)   

            # check
            if len(map_out[sh]) != sum(map_in[sh].values()):
                print('ERROR: number of idcs extracted and number of files in subset of directories do not match for shuffle {}'.format(sh))
    # check: l=list(set([re.search('labeled-data/(.*)/', df.iloc[el].name).group(1)  for el in map_shuffle_id_to_test_AL_idcs[1]]))
    ########################################################
    # Save results as pickle
    with open(output_dirs_pickle_path,'wb') as file:
        pickle.dump([map_shuffle_id_to_train_dirs_w_nfiles,
                    map_shuffle_id_to_test_AL_dirs_w_nfiles,
                    map_shuffle_id_to_test_OOD_dirs_w_nfiles], file)

    with open(output_idcs_pickle_path,'wb') as file:
        pickle.dump([map_shuffle_id_to_train_idcs,
                    map_shuffle_id_to_test_AL_idcs,
                    map_shuffle_id_to_test_OOD_idcs], file)