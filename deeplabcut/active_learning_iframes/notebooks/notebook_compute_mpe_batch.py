## Notebook to explore computation of Multiple Peak Entropy from Liu et al 2017

####################################################################################################
# %%
import os
from pathlib import Path
import pickle

import cv2
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
import scipy.ndimage.filters as filters

from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.util import visualize
from deeplabcut.pose_estimation_tensorflow.predict_videos import GetPosesofFrames
# from deeplabcut.utils.auxfun_videos import imresize

from deeplabcut.pose_estimation_tensorflow.core.predict_multianimal import find_local_peak_indices_skimage, find_local_peak_indices_maxpool_nms


#########################################################################################
# %%
def compute_batch_scmaps_per_frame(cfg, 
                                    dlc_cfg, 
                                    sess, inputs, outputs, 
                                    parent_directory, 
                                    framelist, 
                                    downsampled_img_ny_nx_nc, # 162, 288, 3
                                    batchsize):
    """
    Batchwise prediction of scmap for frame list in directory.
    Based on  deeplabcut.pose_estimation_tensorflow.predict_videos--> GetPosesofFrames
    """
    
    # Read  number of images
    nframes = len(framelist)

    # Parse image downsampled size
    ny, nx, nc = downsampled_img_ny_nx_nc # 162, 288, 3

    # # Read first image's size----
    # OJO! this won't give 'common' downsampled img size if first image is from ChestnutHorseLight (that video is at double resolution)
    # im = imread(os.path.join(parent_directory, 
    #                          framelist[0]), mode="skimage")
    
    # ny, nx, nc = np.shape(im)
    # print("Overall # of frames: ", nframes,
    #       " found with (before cropping) frame dimensions: ", nx,ny)

    # Initialise 
    list_scmaps_per_frame = []
    # batch_scmaps = np.zeros((nframes, 
    #                          nrows_scmap,
    #                          ncols_scmap,
    #                          n_bdprts))

    ## Setup cropping params if required
    if cfg["cropping"]:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
               % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]))
        nx, ny = cfg["x2"] - cfg["x1"], cfg["y2"] - cfg["y1"]
        if nx > 0 and ny > 0:
            pass
        else:
            raise Exception("Please check the order of cropping parameter!")
        if (
            cfg["x1"] >= 0
            and cfg["x2"] < int(np.shape(im)[1])
            and cfg["y1"] >= 0
            and cfg["y2"] < int(np.shape(im)[0])
        ):
            pass  # good cropping box
        else:
            raise Exception("Please check the boundary of cropping!")

    # initialise params for batch processing
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))

    ## if batch=1
    if batchsize == 1:
        for counter, framename in enumerate(framelist):
            
            # progressbar
            if counter != 0 and counter % step == 0:
                pbar.update(step)

            # read img
            im = imread(os.path.join(parent_directory, framename), mode="skimage")
            #---------------------------------------------------
            if im.shape != downsampled_img_ny_nx_nc:
                im = cv2.resize(im, dsize=(nx,ny), interpolation=cv2.INTER_CUBIC)
            #---------------------------------------------------

            # crop if req
            if cfg["cropping"]:
                frame = img_as_ubyte(im[cfg["y1"] : cfg["y2"], 
                                        cfg["x1"] : cfg["x2"], :])
            else:
                frame = img_as_ubyte(im)

            # run inference
            scmap, locref, pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs,
                                                  outall=True)
            list_scmaps_per_frame[counter] = scmap

    ## if batchsize !=1       
    else:
        # initialise array with all frames of batch
        frames = np.empty((batchsize, ny, nx, 3), 
                          dtype="ubyte")  # this keeps all the frames of a batch

        # loop thru frames in full list  and add to batch               
        for counter, framename in enumerate(framelist):
            if counter != 0 and counter % step == 0:
                pbar.update(step)

            im = imread(os.path.join(parent_directory, framename), 
                        mode="skimage")
            #---------------------------------------------------
            if im.shape != downsampled_img_ny_nx_nc:
                im = cv2.resize(im, dsize=(nx,ny), interpolation=cv2.INTER_CUBIC)
            #---------------------------------------------------
            if cfg["cropping"]:
                frames[batch_ind] = img_as_ubyte(im[cfg["y1"] : cfg["y2"], 
                                                    cfg["x1"] : cfg["x2"], :])
            else:
                frames[batch_ind] = img_as_ubyte(im)

            if batch_ind == batchsize - 1:
                scmap, locref, pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs,
                                                        outall=True)
                list_scmaps_per_frame[
                    batch_num * batchsize : (batch_num + 1) * batchsize] = scmap

                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1

        # take care of the last frames (the batch that might have been processed)
        if (batch_ind > 0):  
            # process the whole batch (some frames might be from previous batch!)
            scmap, locref, pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs,
                                                    outall=True)  
            
            list_scmaps_per_frame[
                batch_num * batchsize : batch_num * batchsize + batch_ind] = scmap[:batch_ind, :,:,:]

    pbar.close()
    scmaps_all_frames = np.stack(list_scmaps_per_frame,axis=0)    
    return scmaps_all_frames #, nframes, nx, ny

################################################
def compute_mpe_per_bdprt_and_frame(scmaps_all_frames,
                                    min_px_btw_peaks,
                                    min_peak_intensity,
                                    max_n_peaks):

    # scmap_one_frame = list_scmaps_per_frame[0]
    # all_joints_names = dlc_cfg["all_joints_names"]

    # list_local_max_coordinates_per_frame = [] # per bodypart
    # list_local_max_pvalues_per_frame = []
    # list_local_max_softmax_per_frame = []
    # list_local_max_entropy_per_frame = []

    # for each frame
    loc_max_per_frame_and_bprt = np.empty((scmaps_all_frames.shape[0],
                                            2, # rows and cols
                                            max_n_peaks, # cols
                                            scmaps_all_frames.shape[-1]))
    sftmx_per_frame_and_bprt = np.empty((scmaps_all_frames.shape[0],
                                        max_n_peaks,
                                        scmaps_all_frames.shape[-1]))
    mpe_per_frame_and_bprt = np.empty((scmaps_all_frames.shape[0],
                                      scmaps_all_frames.shape[-1]))
    max_p_per_frame_and_bprt = np.empty((scmaps_all_frames.shape[0],
                                        scmaps_all_frames.shape[-1]))                                  
    for f in range(scmaps_all_frames.shape[0]):
        # for each bdprt
        list_local_max_coordinates = [] # per bodypart
        list_local_max_pvalues = []
        list_local_max_softmax = []
        list_local_max_entropy = []
        for bp in range(scmaps_all_frames.shape[-1]):

            ## compute local max
            # TODO: use nms?
            local_max_coordinates_rc = peak_local_max(scmaps_all_frames[f,:,:,bp],
                                                        min_distance=min_px_btw_peaks,
                                                        threshold_abs=min_peak_intensity,
                                                        exclude_border=False,
                                                        num_peaks=max_n_peaks) 
            # if no local max found: lower threshold to find top n, with n= max_n_peaks?                                           
            # if local_max_coordinates_rc.size==0:
            #     local_max_coordinates_rc = peak_local_max(scmaps_all_frames[f,:,:,bp],
            #                                                 min_distance=min_px_btw_peaks,
            #                                                 threshold_abs=0,
            #                                                 exclude_border=False,
            #                                                 num_peaks=max_n_peaks)                                             
            list_local_max_coordinates.append(local_max_coordinates_rc) #Peaks are the local maxima in a region of 2 * min_distance + 1 
                

            # extract pvalues for each local max
            local_max_pvalues = scmaps_all_frames[f,
                                                local_max_coordinates_rc[:,0],
                                                local_max_coordinates_rc[:,1],
                                                bp] 
            list_local_max_pvalues.append(local_max_pvalues)

            ### compute softmax over local max
            local_max_softmax = softmax(local_max_pvalues, axis=0)
            list_local_max_softmax.append(local_max_softmax)

            ### compute entropy across normalised prob of local max
            list_local_max_entropy.append(entropy(local_max_softmax,axis=0))

        loc_max_per_frame_and_bprt[f,0,:,:] = np.array([x[:,0] for x in list_local_max_coordinates]).T #row
        loc_max_per_frame_and_bprt[f,1,:,:] = np.array([x[:,1] for x in list_local_max_coordinates]).T #col
        sftmx_per_frame_and_bprt[f,:,:] = np.array(list_local_max_softmax).T
        mpe_per_frame_and_bprt[f,:] = list_local_max_entropy
        max_p_per_frame_and_bprt[f,:] = [max(x) for x in list_local_max_pvalues] # get max within local max per bodypart and frame

    return mpe_per_frame_and_bprt, \
            sftmx_per_frame_and_bprt, \
            loc_max_per_frame_and_bprt, \
            max_p_per_frame_and_bprt
#####################################################################################################
# %%
## Inputs 
# Model to run inference on
cfg_path = '/home/sofia/datasets/Horse10_AL_unif_fr/Horse10_AL_unif000/config.yaml'
    # '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/config.yaml'--100k ITERS
    # '/home/sofia/datasets/Horse10_AL_unif_fr/Horse10_AL_unif000/config.yaml'--200k ITERS
shuffle = 1
modelprefix = ''

# Samples
batch_size_inference = 4
downsampled_img_ny_nx_nc = (162, 288, 3)
path_to_h5_file = os.path.join(os.path.dirname(cfg_path),
                             'training-datasets',
                             'iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5')
path_to_pickle_w_base_idcs = '/home/sofia/datasets/Horse10_AL_uncert_OH/horses_AL_OH_train_test_idcs_split.pkl'

# MPE params
min_px_btw_peaks = 2 # Peaks are the local maxima in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`).
min_peak_intensity = 0 #.001 #0.001 # 0.001
max_n_peaks = 5#float('inf')
flag_plot_max_p_per_bdprt = True

# GPU
gpu_to_use = 0
##################################################################################################
# %%
## Load model config
cfg = auxiliaryfunctions.read_config(cfg_path)


## Load test config
trainingsetindex = shuffle-1 # get trainFraction: for these models, one fraction per shuffle
trainFraction = cfg["TrainingFraction"][trainingsetindex] # trainingsetindex = 3
model_folder = os.path.join(cfg["project_path"],
                            str(auxiliaryfunctions.get_model_folder(trainFraction, 
                                                                    shuffle, 
                                                                    cfg, 
                                                                    modelprefix=modelprefix)))

path_test_config = Path(model_folder) / "test" / "pose_cfg.yaml"
dlc_cfg = load_config(str(path_test_config))


## Load snapshot
# Check which snapshots are available and sort them by # iterations
Snapshots = np.array([fn.split(".")[0]
                          for fn in os.listdir(os.path.join(model_folder, "train"))
                          if "index" in fn])
increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

# Use last snapshot
snapshotindex = -1
print("Using %s" % Snapshots[snapshotindex], "for model", model_folder)

# Set ini weights in test config to snapshot wights
dlc_cfg["init_weights"] = os.path.join(model_folder, "train", Snapshots[snapshotindex])


###########################################################################
# %%
## Setup TF graph
# see also: /home/sofia/DeepLabCut/deeplabcut/pose_estimation_tensorflow/core/predict.py

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.compat.v1.reset_default_graph()
dlc_cfg["batch_size"] = batch_size_inference #cfg["batch_size"] # OJO this is batch size for inference
dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg) # pass config loaded, not path, use load_config

##########################################################
# %% 
# Prepare batch of OOD images
# Load train/test base indices from pickle
with open(path_to_pickle_w_base_idcs,'rb') as file:
    # pickle.load(file)
    [map_shuffle_id_to_base_train_idcs,
      map_shuffle_id_to_AL_train_idcs,
      map_shuffle_id_to_OOD_test_idcs]=pickle.load(file)

# Get path for test OOD idcs
df_groundtruth = pd.read_hdf(path_to_h5_file)

# list_test_OOD_images = list(df_groundtruth.index[map_shuffle_id_to_OOD_test_idcs[shuffle]])
list_AL_train_images = list(df_groundtruth.index[map_shuffle_id_to_AL_train_idcs[shuffle]])

######################################################
# %%
# Run inference on batch 
scmaps_all_frames = compute_batch_scmaps_per_frame(cfg, 
                                                    dlc_cfg, 
                                                    sess, inputs, outputs, 
                                                    os.path.dirname(cfg_path), 
                                                    list_AL_train_images, #list_test_OOD_images, 
                                                    downsampled_img_ny_nx_nc,
                                                    batch_size_inference)


#    
###################################################################
# %%  Compute uncertainty per bodypart & per frame
mpe_per_frame_and_bprt, \
    sftmx_per_frame_and_bprt,\
    loc_max_per_frame_and_bprt,\
    max_p_per_frame_and_bprt = compute_mpe_per_bdprt_and_frame(scmaps_all_frames,
                                                                 min_px_btw_peaks,
                                                                 min_peak_intensity,
                                                                 max_n_peaks)
###################################################################                                                                 
# %% Compute mean, max, median  of mpe per frame
mean_mpe_per_frame = np.mean(mpe_per_frame_and_bprt,axis=-1)
max_mpe_per_frame = np.max(mpe_per_frame_and_bprt,axis=-1)
median_mpe_per_frame = np.median(mpe_per_frame_and_bprt,axis=-1)

# sort AL train idcs by mean_mpe
list_AL_train_idcs_ranked_by_mean_mpe =\
    [id for id, mean_mpe in sorted(zip(map_shuffle_id_to_AL_train_idcs[shuffle], #idcs from OOD set
                                       mean_mpe_per_frame),
                                   key=lambda pair: pair[1],
                                   reverse=True)] # sort by the second element of the tuple
list_AL_train_images_sorted_by_mean_mpe = list(df_groundtruth.index[list_AL_train_idcs_ranked_by_mean_mpe])

# sort AL train idcs by mean_mpe
list_AL_train_idcs_ranked_by_max_mpe =\
    [id for id, mean_mpe in sorted(zip(map_shuffle_id_to_AL_train_idcs[shuffle], #idcs from OOD set
                                       max_mpe_per_frame),
                                   key=lambda pair: pair[1],
                                   reverse=True)] # sort by the second element of the tuple
list_AL_train_images_sorted_by_max_mpe = list(df_groundtruth.index[list_AL_train_idcs_ranked_by_max_mpe])
###################################################################
# %% plot results per frame
mean_max_p_per_frame = np.mean(max_p_per_frame_and_bprt,axis=-1)

fig, ax1 = plt.subplots()
ax1.plot(mean_max_p_per_frame,
        color='tab:orange')
ax1.set_ylabel('max p', color='tab:orange')
ax1.set_ylim([0.55,1.05])

ax2=ax1.twinx()
ax2.plot(mean_mpe_per_frame,
        color='tab:blue')
# ax2.plot(max_mpe_per_frame,'.',
#         color='tab:red')
ax2.set_ylabel('MPE', color='tab:blue')
plt.show()

## plot mean, max, median mpe per frame vs max p per frame
plt.plot(mean_max_p_per_frame,mean_mpe_per_frame,'.',color='tab:green',label='mean MPE')
plt.plot(mean_max_p_per_frame,max_mpe_per_frame,'.',color='tab:red',label='max MPE')
plt.plot(mean_max_p_per_frame,median_mpe_per_frame,'.',color='tab:blue', label='median MPE')
plt.vlines(0.5, 1.5, 1.62,linestyles='--')
plt.legend()
plt.xlabel('mean max p')
plt.ylabel('MPE')
plt.show()
# ---
# local_max_coordinates_rc = find_local_peak_indices_maxpool_nms(scmaps_all_frames, 
#                                                                 min_px_btw_peaks, 
#                                                                 min_peak_intensity)
# out = local_max_coordinates_rc.eval(session=tf.compat.v1.Session())     # shape=(465059, 4)---why 4?? top 2 peaks?

# local_max_coordinates_rc = find_local_peak_indices_skimage(scmaps_all_frames,
#                                                             min_px_btw_peaks, 
#                                                             min_peak_intensity)

###################################################################
# %% Check results same as before....
frame_id = 25

list_local_max_coordinates = []
list_local_max_pvalues = []
list_local_max_softmax = []
list_local_max_entropy = []
for bp in range(scmaps_all_frames.shape[-1]):
    local_max_coordinates_rc = peak_local_max(scmaps_all_frames[frame_id,:,:,bp],
                                                min_distance=min_px_btw_peaks,
                                                threshold_abs=min_peak_intensity,
                                                exclude_border=False,
                                                num_peaks=max_n_peaks) 
    list_local_max_coordinates.append(local_max_coordinates_rc)

    local_max_pvalues = scmaps_all_frames[frame_id,
                                        local_max_coordinates_rc[:,0],
                                        local_max_coordinates_rc[:,1],
                                        bp] 
    list_local_max_pvalues.append(local_max_pvalues)

    ### Softmax and entropy
    local_max_softmax = softmax(local_max_pvalues, axis=0)
    list_local_max_softmax.append(local_max_softmax)
    ### Entropy
    list_local_max_entropy.append(entropy(local_max_softmax,axis=0))       

# Compare
if not (np.abs(mpe_per_frame_and_bprt[25,:] - np.array(list_local_max_entropy)) \
    < 1e-06).all():
    print('mismatch in mpe calculation')     
else:
    print('mpe matches')
if not (np.abs(sftmx_per_frame_and_bprt[25,:] - np.array(list_local_max_softmax).T) < 1e-06).all():
    print('mismatcg in softmax calculation')
else:
    print('softmax matches')
if not (loc_max_per_frame_and_bprt[25,:,:,:].T == np.array(list_local_max_coordinates)).all():
    print('mismatch in local max row and col calculation')
else:
    print('local max coords matches')

if not (np.abs(np.array([max(x) for x in list_local_max_pvalues]) \
        - max_p_per_frame_and_bprt[25,:]) < 1e-6).all():
    print('mismatch in max p per frame calculation')
else:
    print('max p per frame matches')


###################################################################
# %%  Plot uncertainty score per frame and bodypart

# # scmap_one_frame = list_scmaps_per_frame[0]
# all_joints_names = dlc_cfg["all_joints_names"]

# list_local_max_coordinates = [] # per bodypart
# list_local_max_pvalues = []
# list_local_max_softmax = []
# list_local_max_entropy = []

# # for each frame
# for f in [25]:#range(scmaps_all_frames.shape[0]):
#     # for each bdprt
#     for bp in range(scmaps_all_frames.shape[-1]):
#         local_max_coordinates_rc = peak_local_max(scmaps_all_frames[f,:,:,bp],
#                                                     min_distance=min_px_btw_peaks,
#                                                     threshold_abs=min_peak_intensity,
#                                                     exclude_border=False,
#                                                     num_peaks=max_n_peaks) 
                                        
#         list_local_max_coordinates.append(local_max_coordinates_rc) #Peaks are the local maxima in a region of 2 * min_distance + 1 
            

#         # pvalues
#         local_max_pvalues = scmaps_all_frames[f,
#                                               local_max_coordinates_rc[:,0],
#                                               local_max_coordinates_rc[:,1],
#                                               bp] 
#         list_local_max_pvalues.append(local_max_pvalues)

#         ### Softmax and entropy
#         local_max_softmax = softmax(local_max_pvalues, axis=0)
#         list_local_max_softmax.append(local_max_softmax)
#         ### Entropy
#         list_local_max_entropy.append(entropy(local_max_softmax,axis=0))
#         #----------
#         # plot 
#         plt.matshow(scmaps_all_frames[f,:,:,bp])
#         plt.scatter(local_max_coordinates_rc[:,1],
#                     local_max_coordinates_rc[:,0],
#                     10,'r')
#         for k in range(local_max_coordinates_rc.shape[0]):
#             plt.text(local_max_coordinates_rc[k,1],
#                     local_max_coordinates_rc[k,0],
#                     '{:.2f}'.format(local_max_pvalues[k]),
#                     fontsize=12,
#                     fontdict={'color':'r'})            
#         plt.title('{} - local maxima and confidence (MPE={:.2f})'.format(all_joints_names[bp],
#                                                                         list_local_max_entropy[-1]))    
#         plt.show() 

#         plt.plot(local_max_softmax,'.-')
#         plt.title('{} - normalised prob across maxima (MPE={:.2f})'.format(all_joints_names[bp], 
#                                                                         list_local_max_entropy[-1]))
#         plt.xlabel('local maximum id')
#         plt.ylabel('normalised prob')
#         plt.show()
#         #---------

#     #---------------------------------------
#     ## Compate MPE and p metrics
#     fig, ax1 = plt.subplots()

#     # MPE in ax1
#     ax1.plot(list(range(len(list_local_max_entropy))),
#             list_local_max_entropy,
#             '.-',
#             color='tab:blue')   
#     ax1.set_xticks(list(range(len(list_local_max_entropy))),
#                     all_joints_names,
#                     rotation = 90,
#                     ha='center')
#     ax1.set_ylabel('MPE', color='tab:blue')
#     # ax1.set_ylim([-0.05,2])

#     # optionally: max p per bodypart in ax2
#     if flag_plot_max_p_per_bdprt:
#         ax2 = ax1.twinx()
#         # p value per bdprt in ax2         
#         ax2.plot(list(range(len(list_local_max_entropy))),
#                 [np.max(x) for x in list_local_max_pvalues],
#                 '.-',
#                 color='tab:orange')          
#         ax2.set_ylabel('p', color='tab:orange')               
#     plt.title('MPE/max p per bodypart')         
#     plt.show()


