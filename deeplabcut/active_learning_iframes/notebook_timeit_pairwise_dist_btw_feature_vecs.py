
# %% time pdist torch approach
import timeit

torch_pdist_setup = '''
from deeplabcut.active_learning_iframes.infl_horse_dataset_utils import pairwise_cosine_distance_two_inputs
import pickle

path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_test_idcs_split.pkl' 
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [map_shuffle_id_to_base_train_idcs,
        map_shuffle_id_to_AL_train_idcs,
        map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)

sh = 1
query_idcs = map_shuffle_id_to_AL_train_idcs[sh]
ref_idcs = map_shuffle_id_to_AL_train_idcs[sh]

with open('feature_tensors.pkl','rb') as file:
    feature_tensors = pickle.load(file)

'''
torch_pdist_stmt = 'pairwise_cosine_distance_two_inputs(feature_tensors[query_idcs,:],feature_tensors[ref_idcs,:])'
torch_pdist_n_exec = 1000
print('Time per execution = {} s'.format(timeit.timeit(stmt = torch_pdist_stmt,
                                                     setup = torch_pdist_setup,
                                                     number = torch_pdist_n_exec)/torch_pdist_n_exec))

# %% time cdist approach
import timeit

cdist_setup = '''
from scipy.spatial.distance import cdist
import pickle

path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_OH_outputs/horses_AL_OH_train_test_idcs_split.pkl' 
with open(path_to_pickle_w_base_idcs,'rb') as file:
    [map_shuffle_id_to_base_train_idcs,
        map_shuffle_id_to_AL_train_idcs,
        map_shuffle_id_to_OOD_test_idcs] = pickle.load(file)

sh = 1
query_idcs = map_shuffle_id_to_AL_train_idcs[sh]
ref_idcs = map_shuffle_id_to_AL_train_idcs[sh]

with open('feature_arrays.pkl','rb') as file:
    feature_arrays = pickle.load(file)

'''
cdist_stmt = 'cdist(feature_arrays[query_idcs,:], feature_arrays[ref_idcs,:], "cosine")'
cdist_n_exec = 2
print('Time per execution = {} s'.format(timeit.timeit(stmt = cdist_stmt,
                                                     setup = cdist_setup,
                                                     number = cdist_n_exec)/cdist_n_exec))
# print(timeit.timeit(stmt = cdist_str,
#                     setup = cdist_setup,
#                     number = 5))
# %%
