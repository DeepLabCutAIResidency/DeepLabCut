## Notebook to explore computation of Multiple Peak Entropy from Liu et al 2017

# %%
import os
from pathlib import Path

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict

import tensorflow as tf

import cv2
import numpy as np
from skimage.util import img_as_ubyte

# import argparse
# import os
# import os.path
# import pickle
# import re
# import time
# import warnings
# from pathlib import Path


# import pandas as pd
# 
# from scipy.optimize import linear_sum_assignment
# from skimage.util import img_as_ubyte
# from tqdm import tqdm

# from deeplabcut.pose_estimation_tensorflow.config import load_config
# from deeplabcut.pose_estimation_tensorflow.core import predict
# from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils

# from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
# from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
# from deeplabcut.pose_estimation_tensorflow.core.openvino.session import (
#     GetPoseF_OV,
#     is_openvino_available,
# )


############################################
# %%
## Inputs 
cfg_path = '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/config.yaml'
shuffle = 1
modelprefix = ''

frame_path = '/home/sofia/datasets/Horse10_AL_unif/Horse10_AL_unif000/labeled-data/Sample2/0369.png'

gpu_to_use = 1
######################################
# %%
## Get model config params
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

###############################################
# Check which snapshots are available and sort them by # iterations
try:
    Snapshots = np.array(
        [
            fn.split(".")[0]
            for fn in os.listdir(os.path.join(model_folder, "train"))
            if "index" in fn
        ]
    )
except FileNotFoundError:
    raise FileNotFoundError(
        "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
        % (shuffle, shuffle)
    )

if cfg["snapshotindex"] == "all":
    print(
        "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
    )
    snapshotindex = -1
else:
    snapshotindex = cfg["snapshotindex"]

increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print("Using %s" % Snapshots[snapshotindex], "for model", model_folder)

############################################
# %%
# Check if data already was generated:
dlc_cfg["init_weights"] = os.path.join(model_folder, "train", Snapshots[snapshotindex])

# update batchsize for inference (based on parameters in config.yaml)
dlc_cfg["batch_size"] = 1 #cfg["batch_size"] # OJO this is batch size for inference
#############################################
# %%
## Setup TF graph
# see also: /home/sofia/DeepLabCut/deeplabcut/pose_estimation_tensorflow/core/predict.py

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.compat.v1.reset_default_graph()

sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg) # pass config loaded, not path, use load_config

# update number of outputs and adjust pandas indices
dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)

# ----
# Check if data already was generated:
# dlc_cfg["init_weights"] = os.path.join(modelfolder, 
#                                       "train", 
#                                       Snapshots[snapshotindex])
# trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]

# update batchsize (based on parameters in config.yaml)
# dlc_cfg["batch_size"] = cfg["batch_size"]

# # Name for scorer:
# DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(cfg,
#     shuffle,
#     trainFraction,
#     trainingsiterations=trainingsiterations,
#     modelprefix=modelprefix,
# )

# sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg) # pass config loaded, not path, use load_config


##################################################
# %%
# Run inference---eventually in a batch
im = imread(frame_path, mode="skimage")
frame = img_as_ubyte(im)

# src = cv2.imread(frame_path)
# frame = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# frame = img_as_ubyte(frame)

# img = np.expand_dims(frame, axis=0).astype(float)
# outputs_np = sess.run(outputs, feed_dict={inputs: img})
# scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
# num_outputs = cfg.get("num_outputs", 1)

scmap, locref, pose = predict.getpose(frame, #np array
                                        dlc_cfg, 
                                        sess, 
                                        inputs, 
                                        outputs,
                                        outall=True) #getpose(image, cfg, sess, inputs, outputs, outall=False)

# scmap.shape   (22, 36, 22)
# locref       (22, 36, 22, 2)       
# pose.shape       (22, 3)            
# %%
