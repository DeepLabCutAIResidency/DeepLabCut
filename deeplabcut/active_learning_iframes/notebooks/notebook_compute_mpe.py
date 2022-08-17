## Notebook to explore computation of Multiple Peak Entropy from Liu et al 2017

####################################################################################################
# %%
import os
from pathlib import Path



import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
import scipy.ndimage.filters as filters

import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.feature import peak_local_max

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.util import visualize
# from deeplabcut.utils.auxfun_videos import imresize

from deeplabcut.pose_estimation_tensorflow.core.predict_multianimal import find_local_peak_indices_maxpool_nms

#####################################################################################################
# %%
## Inputs 
cfg_path = '/home/sofia/datasets/Horse10_AL_unif_fr/Horse10_AL_unif000/config.yaml'
    #'/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/config.yaml'#
shuffle = 1
modelprefix = ''

frame_path = '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/labeled-data/ChestnutHorseLight/0243.png'

gpu_to_use = 1
##################################################################################################
# %%
## Get model params
# get trainFraction: for these models, one fraction per shuffle
cfg = auxiliaryfunctions.read_config(cfg_path)
trainingsetindex = shuffle-1
trainFraction = cfg["TrainingFraction"][trainingsetindex] # trainingsetindex = 3

# get test config
model_folder = os.path.join(cfg["project_path"],
                            str(auxiliaryfunctions.get_model_folder(trainFraction, 
                                                                    shuffle, 
                                                                    cfg, 
                                                                    modelprefix=modelprefix)))

path_test_config = Path(model_folder) / "test" / "pose_cfg.yaml"
dlc_cfg = load_config(str(path_test_config))

##########################################################
# %%
## Get snapshot
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
# update batchsize for inference 
dlc_cfg["batch_size"] = 1 #cfg["batch_size"] # OJO this is batch size for inference

sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg) # pass config loaded, not path, use load_config

# update number of outputs and adjust pandas indices
# dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)


######################################################
# %%
# Run inference---eventually in a batch
im = imread(frame_path, mode="skimage")
frame = img_as_ubyte(im) #---- is frame passed ok, ok size?

# src = cv2.imread(frame_path)
# frame = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# frame = img_as_ubyte(frame)

# img = np.expand_dims(frame, axis=0).astype(float)
# outputs_np = sess.run(outputs, feed_dict={inputs: img})
# scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
# num_outputs = cfg.get("num_outputs", 1)

scmap, locref, pose = predict.getpose(frame, #np array # (162, 288, 3)
                                        dlc_cfg, 
                                        sess, 
                                        inputs, 
                                        outputs,
                                        outall=True) #getpose(image, cfg, sess, inputs, outputs, outall=False)

# scmap.shape   (22, 36, 22) --last dimension is joint
# locref       (22, 36, 22, 2)       
# pose.shape       (22, 3)           
###################################################################
# %%  Compute uncertainty score per bodypart

### Extract model's confidence in local max
neighborhood_size = 5 #
# find_local_peak_indices_maxpool_nms(scmap, neighborhood_size, np.min(find_local_peak_indices_maxpool_nms))

list_local_max_coordinates = []
list_local_max_pvalues = []
list_local_max_softmax = []
list_local_max_entropy = []
for j in range(scmap.shape[-1]):
    # local max
    local_max_coordinates = peak_local_max(scmap[:,:,j],min_distance=5) #---why no result if mindist=7?
    # local_max_coordinates = find_local_peak_indices_maxpool_nms(scmap[:,:,j], 
                                                                # neighborhood_size, 
                                                                # np.min(find_local_peak_indices_maxpool_nms))

    list_local_max_coordinates.append(local_max_coordinates) #Peaks are the local maxima in a region of 2 * min_distance + 1 
    
    # pvalues
    local_max_pvalues = scmap[local_max_coordinates[:,0],
                              local_max_coordinates[:,1],j] 
    list_local_max_pvalues.append(local_max_pvalues)


    ### Softmax
    local_max_softmax = softmax(local_max_pvalues, axis=0)
    list_local_max_softmax.append(local_max_softmax)


    ### Entropy
    list_local_max_entropy.append(entropy(local_max_softmax,axis=0))


###################################################################
# %% Visualise results
# Based on visualize.show_heatmaps(dlc_cfg, frame, scmap, pose)
# visualize.waitforbuttonpress()
import cv2
plt_interp = "bilinear"
# all_joints = dlc_cfg["all_joints"] # starts with 0
all_joints_names = dlc_cfg["all_joints_names"]
cmap="jet"

idcs_sorted_by_entropy = sorted(range(len(list_local_max_entropy)), 
                                key=lambda k: list_local_max_entropy[k])

for pidx, idx_s in enumerate(idcs_sorted_by_entropy):

    ## get heatmap for this bodypart
    scmap_part = scmap[:, :, idx_s].squeeze() #if batch:L sum over last dim (batch)?

    # resize: cv2.resize fx and fy = 8 (bc  stride=8?) should it be dlc_cfg['stride']?
    # scmap_part = imresize(scmap_part, 8.0, interpolationmethod=1) #---interp="bicubic"

    scmap_part = cv2.resize(scmap_part, #None, fx=size, fy=size, 
                            (im.shape[1],im.shape[0]),
                            interpolation=cv2.INTER_CUBIC) #'bicubic

    # scmap_part = resize(scmap_part, 
    #                     im.shape) #(shape_dest[0], shape_dest[1]))

    # scmap_part = np.lib.pad(scmap_part, 
    #                         ((4, 0), (4, 0)), 
    #                         "minimum") # (180, 292)

    # plot heatmap
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()

    ax.imshow(frame) #, interpolation=plt_interp)
    hm=ax.imshow(scmap_part, 
                  alpha=0.5, 
                  cmap=cmap, 
                  interpolation=plt_interp)
    hm.set_clim(vmin=0,vmax=1)     
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(hm, cax=cax)

    # plot relevant keypoint in red
    ax.scatter(pose[idx_s,0],
               pose[idx_s,1],
                10,
                'r',
                'x')

    ax.set_title('{} - {} (p={:.3f}, MPE={:.3f}'.format(idx_s,
                                                        all_joints_names[idx_s],
                                                        pose[idx_s,2],
                                                        list_local_max_entropy[idx_s]))            
    plt.show()

# plot bodyparts
# plt.figure(figsize=(10,10))
# plt.imshow(visualize.visualize_joints(frame, pose)) # (162, 288, 3)
# plt.show()

# curr_plot = axarr[0, 0]


# %%
plt.plot(list_local_max_entropy)
plt.plot(pose[:,2])


# %%
# # For every img
# #   For every bodypart
# #    Extract local max
# neighborhood_size = 5
# scmap_local_max = filters.maximum_filter(scmap, 
#                                          neighborhood_size,
#                                          mode='reflect')


# # plot                                        
# list_joints = dlc_cfg["all_joints_names"]
# cmap="jet"
# plt_interp = "bilinear"
# scmap_up_local_max = np.empty((im.shape[0],im.shape[1],scmap.shape[-1]))
# for bp_i, bp_str in enumerate(list_joints):
#     fig = plt.figure(figsize=(10,10))
#     ax = plt.gca()

#     # get heatmap for this bodypart
#     scmap_part = scmap[:, :, bp_i].squeeze()

#     # upsample
#     scmap_part_upsampled = cv2.resize(scmap_part, #None, fx=size, fy=size, 
#                                     (im.shape[1],im.shape[0]),
#                                     interpolation=cv2.INTER_CUBIC)
#     # pass maxpool filter                                
#     scmap_up_local_max[:,:,bp_i] = filters.maximum_filter(scmap_part_upsampled, 
#                                                             neighborhood_size,
#                                                             mode='reflect') #default

#     ax.imshow(frame) #, interpolation=plt_interp)
#     hm=ax.imshow(scmap_up_local_max[:,:,bp_i], #scmap_part_upsampled, #
#                   alpha=0.5, 
#                   cmap=cmap, 
#                   interpolation=plt_interp)
#     # hm.set_clim(vmin=0,vmax=1)     
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05) 
#     plt.colorbar(hm, cax=cax)   

#     ax.set_title('{} - {} (p={:.3f})'.format(bp_i,
#                                             bp_str,
#                                             pose[bp_i,2]))    
#     plt.show()           

# # # Extract local maxima 
# # list_joints = dlc_cfg["all_joints_names"]
# # neighborhood_size = 5
# # scmap_local_max = np.empty_like(scmap.shape)
# # for bp_i, bp_str in enumerate(list_joints):
# #     bp_i
# #     scmap_part = scmap[:, :, bp_i].squeeze()
# #     # scmap_part_upsampled = resize(scmap_part, #list_jointsNone, fx=size, fy=size, 
# #     #                               im.shape),
# #                                       #interpolation=cv2.INTER_CUBIC)
# #     scmap_local_max[:,:,bp_i-1] = filters.maximum_filter(scmap_part, 
# #                                                          neighborhood_size,
# #                                                          mode='reflect') #default

# # Compute normalised prediction for each local maxima



# # Compute Multiple Peak Entropy uncertainty of bodypart p in image I


# %%
