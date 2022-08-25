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


###############################################################
def set_inference_params_in_test_cfg(cfg_path,
                                        batch_size_inference,
                                        shuffle,
                                        trainingsetindex = None, # if None, extracted from config based on shuffle number
                                        modelprefix='',
                                        snapshotindex=-1): # idx of snapshot sorted by increasing number, typically snapshotindex=-1 to get last one

    ## Load model config
    cfg = auxiliaryfunctions.read_config(cfg_path)

    ## Load test config
    if trainingsetindex==None:
        trainingsetindex = shuffle-1 # get trainFraction: for AL models, one fraction per shuffle
    trainFraction = cfg["TrainingFraction"][trainingsetindex] # trainingsetindex = 3
    model_folder = os.path.join(cfg["project_path"],
                                str(auxiliaryfunctions.get_model_folder(trainFraction, 
                                                                        shuffle, 
                                                                        cfg, 
                                                                        modelprefix=modelprefix)))
    path_test_config = Path(model_folder) / "test" / "pose_cfg.yaml"
    dlc_cfg = load_config(str(path_test_config))


    ## Locate desired snapshot
    # Check which snapshots are available and sort them by # iterations
    Snapshots = np.array([fn.split(".")[0]
                            for fn in os.listdir(os.path.join(model_folder, "train"))
                            if "index" in fn])
    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    # print snapshot used
    print("Using %s" % Snapshots[snapshotindex], "for model", model_folder)

    ## Set ini weights in test config to point to selected snapshot weights
    dlc_cfg["init_weights"] = os.path.join(model_folder, "train", Snapshots[snapshotindex])

    ## Set batchsize and number of outputs
    dlc_cfg["batch_size"] = batch_size_inference #cfg["batch_size"] 
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

    return dlc_cfg, cfg

########################################################################################
def setup_TF_graph_for_inference(test_pose_cfg, #dlc_cfg: # pass config loaded, not path (use load_config())
                                 gpu_to_use):

    # see also: /home/sofia/DeepLabCut/deeplabcut/pose_estimation_tensorflow/core/predict.py

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tf.compat.v1.reset_default_graph()

    sess, inputs, outputs = predict.setup_pose_prediction(test_pose_cfg) 

    return sess, inputs, outputs

#########################################################################################
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

    ## Setup cropping params if required----croppin relative to donwsampled img right?
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
    return scmaps_all_frames 

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


def compute_mpe_per_frame(cfg,
                          dlc_cfg,
                          sess, inputs, outputs,
                          parent_directory, #cfg_path_for_uncert_snapshot,
                          list_AL_train_images,
                          downsampled_img_ny_nx_nc,
                          batch_size_inference,
                          min_px_btw_peaks,
                          min_peak_intensity,
                          max_n_peaks):
    '''
    Compute scormaps per bdprt and frame, and MPE per bdprt and frame
    - Run inference on AL train images
    - Compute uncertainty (MPE) per bodypart and frame
    - Compute mean, median and max per frame

    '''
    # Run inference on AL train images
    scmaps_all_frames = compute_batch_scmaps_per_frame(cfg, 
                                                        dlc_cfg, 
                                                        sess, inputs, outputs, 
                                                        parent_directory,
                                                        list_AL_train_images, 
                                                        downsampled_img_ny_nx_nc,
                                                        batch_size_inference)

    # Compute uncertainty (MPE) per bodypart and mean/max per frame
    mpe_per_frame_and_bprt, \
    sftmx_per_frame_and_bprt,\
    loc_max_per_frame_and_bprt,\
    max_p_per_frame_and_bprt = compute_mpe_per_bdprt_and_frame(scmaps_all_frames,
                                                                 min_px_btw_peaks,
                                                                 min_peak_intensity,
                                                                 max_n_peaks)
    # compute mean, max and median over all bodyparts per frame                                                             
    mpe_metrics_per_frame = {'mean': np.mean(mpe_per_frame_and_bprt,axis=-1),
                             'max': np.max(mpe_per_frame_and_bprt,axis=-1),
                             'median': np.median(mpe_per_frame_and_bprt,axis=-1)}     

    return mpe_metrics_per_frame, mpe_per_frame_and_bprt