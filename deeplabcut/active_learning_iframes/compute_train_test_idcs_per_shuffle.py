# Min sum partition problem 
# from https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/
"""




"""
import os
import sys
import pickle
import pandas as pd
import re

#####################################################
# Function to find the minimum sum recursivelly
def findMinRec(i,list_S_total,list_S1):
 
    sumTotal = sum(list_S_total)
    sumS1 = sum(list_S1)
    # If we have reached last element.
    # Sum of one subset is sumCalculated,
    # sum of other subset is sumTotal-
    # sumCalculated.  Return absolute
    # difference of two sums.
    if (i == 0):
        return (abs((sumTotal - sumS1) - sumS1), list_S1)
 
    # For every item arr[i], we have two choices
    # (1) We do not include it first set
    # (2) We include it in first set
    # We return minimum of two choices
    list_S1_added = list_S1.copy()
    list_S1_added.append(list_S_total[i - 1])
    return min(findMinRec(i-1,
                          list_S_total, 
                          list_S1_added),
               findMinRec(i-1,
                          list_S_total, 
                          list_S1))
 
##################################################### 
def findMin(list_S_total):

    # Compute initial sum in set S1
    list_S1_ini = []
     
    # Compute result using
    # recursive function
    return findMinRec(len(list_S_total), # i
                      list_S_total, 
                      list_S1_ini)

###################################################
def convert_dir_list_to_dict_w_nfiles(list_dirs,
                                      parent_dir_path):
    # Returns dict in which keys are the dirs in the input list and values are the number of png files in that dir                                   
    map_dirs_to_nfiles = dict()
    for d in list_dirs:
        num_files_in_d = len([f for f in os.listdir(os.path.join(parent_dir_path,d)) if f.endswith('.png')])
        map_dirs_to_nfiles.update({d:num_files_in_d})
    return map_dirs_to_nfiles

########################################################
def compute_train_test_dirs_dicts(parent_dir_path,
                                  map_shuffle_id_to_train_dirs):
    """
    Returns three dicts:
    - map_horses_dirs_to_nfiles
    - map_shuffle_id_to_train_dirs_w_nfiles
    - map_shuffle_id_to_test_dirs_w_nfiles
    
    """                              
    ########################################
    # Get all horses directories and their number of files 
    list_all_dirs = [el for el in os.listdir(parent_dir_path) if os.path.isdir(os.path.join(parent_dir_path,el))] # only parent dirs (30)
    list_all_dirs.sort() # sort just for convenience
    if len(list_all_dirs) != 30:
        print('ERROR: not exactly 30 horses in dataset')
        sys.exit(1)

    # convert to dict with nfiles per dir
    map_horses_dirs_to_nfiles = convert_dir_list_to_dict_w_nfiles(list_all_dirs,
                                                                  parent_dir_path)

    ################################################################
    # Compute dict with train directories per shuffle and number of files
    if [len(map_shuffle_id_to_train_dirs[ky]) for ky in [1,2,3]]!=[10,10,10]:
        print('ERROR: not exactly 10 horses per shuffle in train set')
        sys.exit(1)

    # Convert lists to dicts with number of files per dir
    n_shuffles = len(map_shuffle_id_to_train_dirs.keys())
    map_shuffle_id_to_train_dirs_w_nfiles = dict()
    for sh in range(1,n_shuffles+1):
        map_shuffle_id_to_train_dirs_w_nfiles[sh] = convert_dir_list_to_dict_w_nfiles(map_shuffle_id_to_train_dirs[sh],
                                                                                      parent_dir_path)
    ################################################################
    ## Compute list of directories in test set (AL+OOD) and convert to dict with nfiles per dir
    map_shuffle_id_to_test_dirs_w_nfiles = dict()
    for sh in range(1,n_shuffles+1):
        list_test_dirs_for_one_shuffle = [el for el in list_all_dirs if el not in map_shuffle_id_to_train_dirs[sh]]
        map_shuffle_id_to_test_dirs_w_nfiles[sh] = convert_dir_list_to_dict_w_nfiles(list_test_dirs_for_one_shuffle,
                                                                                      parent_dir_path)

    return (map_horses_dirs_to_nfiles,
            map_shuffle_id_to_train_dirs_w_nfiles,
            map_shuffle_id_to_test_dirs_w_nfiles)

######################################################
def split_test_set_dicts(map_shuffle_id_to_test_dirs_w_nfiles):
    '''
    
    '''
    map_shuffle_id_to_test_AL_dirs_w_nfiles = dict()            
    map_shuffle_id_to_test_OOD_dirs_w_nfiles = dict()     

    n_shuffles = len(map_shuffle_id_to_test_dirs_w_nfiles.keys())                            
    for sh in range(1,n_shuffles+1):      

        ## Get list of values (i.e.,number of files) for this shuffle (assuming extracted in same order)
        list_test_nfiles_one_shuffle = list(map_shuffle_id_to_test_dirs_w_nfiles[sh].values())
        list_test_dirs_one_shuffle = list(map_shuffle_id_to_test_dirs_w_nfiles[sh].keys())

        ## Compute split between sets of number of files st diff between sets is min
        # also returns list of number of files for one set
        min_diff, list_S1 = findMin(list_test_nfiles_one_shuffle)
        list_S2 = list_test_nfiles_one_shuffle.copy() 
        for el in list_S1:
            list_S2.remove(el) # remove first appearance of element 'el' in S2
        # checks
        if sum(list_S1)+sum(list_S2)!=sum(list_test_nfiles_one_shuffle):
            print('ERROR: sum of elements in subsets does not add up total')
            sys.exit(1)
        if min_diff != abs(sum(list_S1)-sum(list_S2)):
            print('ERROR: mismatch in min diff between sets')
            sys.exit(1)
        # cannot do line below because directories that belong to diff subsets may have same number of files! 
        # list_S2 = [el for el in list_nfiles_one_shuffle if el not in list_S1] 

        ## Get directories names for computed subsets and add to dicts
        map_shuffle_id_to_test_AL_dirs_w_nfiles[sh] = dict()
        for val in list_S1: # find first key with that value
            idx_first_occ = list_test_nfiles_one_shuffle.index(val) # find first occur of that value in list of values
            ky_first_occ = list_test_dirs_one_shuffle[idx_first_occ] # get key of that first occurence
            map_shuffle_id_to_test_AL_dirs_w_nfiles[sh].update({ky_first_occ: val}) #map_shuffle_id_to_test_dirs_w_nfiles[sh][ky_first_occ]})

        map_shuffle_id_to_test_OOD_dirs_w_nfiles[sh] = {ky: map_shuffle_id_to_test_dirs_w_nfiles[sh][ky]
                                                        for ky in map_shuffle_id_to_test_dirs_w_nfiles[sh].keys()
                                                        if ky not in map_shuffle_id_to_test_AL_dirs_w_nfiles[sh].keys()}
        ## Check no overlap between AL and OOD     
        intersection_set = set(map_shuffle_id_to_test_OOD_dirs_w_nfiles[1].keys()).intersection(set(map_shuffle_id_to_test_AL_dirs_w_nfiles[1].keys()))
        if len(intersection_set)!=0:
            print('ERROR: overlap exists between directories in AL test set and OOD test set')                                        

        ## Check all good
        if sum(map_shuffle_id_to_test_AL_dirs_w_nfiles[sh].values())!=sum(list_S1):
            print('ERROR: mismatch in number of files in AL test set')
            sys.exit(1)
        if sum(map_shuffle_id_to_test_OOD_dirs_w_nfiles[sh].values())!=sum(list_S2):
            print('ERROR: mismatch in number of files in OOD test set')
            sys.exit(1)
        

        ## Print results per shuffle
        print('---------------------------')
        print('SHUFFLE {}'.format(sh))
        print("Minimum difference between the two test sets: {}".format(min_diff))
        print('Elements in AL test set:')
        [print('\t {}'.format(el)) for el in map_shuffle_id_to_test_AL_dirs_w_nfiles[sh].keys()] #.format(map_shuffle_id_to_test_AL_dirs_w_nfiles[sh].keys())) #list_S1))
        # print('Elements in S1: {}'.format([dict_n_files_per_dir_per_shuffle[sh][el] for el in list_S1]))
        print('Elements in OOD test set:')
        [print('\t {}'.format(el)) for el in map_shuffle_id_to_test_OOD_dirs_w_nfiles[sh].keys()] #list_S2))
        # print(S1_final)
        print('---------------------------')

    return(map_shuffle_id_to_test_AL_dirs_w_nfiles,
           map_shuffle_id_to_test_OOD_dirs_w_nfiles)

################################################
def compute_idcs_from_list_dirs(list_dirs,
                                df):
    list_parent_dir_per_image_in_df = [re.search('labeled-data/(.*)/', el).group(1) 
                                       for el in df.index.to_list()]
    
    list_image_idcs_in_df = [i for i,p in enumerate(list_parent_dir_per_image_in_df) 
                               if p in list_dirs]

    # list_image_idcs_in_df = []
    # for i,p in enumerate(list_parent_dir_per_image_in_df):
    #     if p in list_dirs:
    #        list_image_idcs_in_df.append(i)
    return list_image_idcs_in_df

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
     map_shuffle_id_to_test_dirs_w_nfiles) = compute_train_test_dirs_dicts(parent_dir_path,
                                                                            map_shuffle_id_to_train_dirs)
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