## Notebook to explore computation of Multiple Peak Entropy from Liu et al 2017

#############################################
# %%
import os
from pathlib import Path

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict

import tensorflow as tf
from scipy.special import softmax
import math
import matplotlib.pyplot as plt

import cv2
import numpy as np
from skimage.util import img_as_ubyte

from deeplabcut.pose_estimation_tensorflow.util import visualize
from deeplabcut.utils.auxfun_videos import imresize


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

# get snapshot
# Check which snapshots are available and sort them by # iterations
Snapshots = np.array([fn.split(".")[0]
                          for fn in os.listdir(os.path.join(model_folder, "train"))
                          if "index" in fn])
increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

if cfg["snapshotindex"] == "all":
    print("Snapshotindex is set to 'all' in the config.yaml file. \
          Running video analysis with all snapshots is very costly! \
          Use the function 'evaluate_network' to choose the best the snapshot. \
          For now, changing snapshot index to -1!")
    snapshotindex = -1
else:
    snapshotindex = cfg["snapshotindex"]

print("Using %s" % Snapshots[snapshotindex], "for model", model_folder)

######################################################################
# %%
# Set ini weights in test config  to snapshot
dlc_cfg["init_weights"] = os.path.join(model_folder, "train", Snapshots[snapshotindex])

# update batchsize for inference 
dlc_cfg["batch_size"] = 1 #cfg["batch_size"] # OJO this is batch size for inference

###########################################################################
# %%
## Setup TF graph
# see also: /home/sofia/DeepLabCut/deeplabcut/pose_estimation_tensorflow/core/predict.py

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.compat.v1.reset_default_graph()

sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg) # pass config loaded, not path, use load_config

# update number of outputs and adjust pandas indices
# dlc_cfg["num_outputs"] = cfg.get("num_outputs", 1)


######################################################
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

scmap, locref, pose = predict.getpose(frame, #np array # (162, 288, 3)
                                        dlc_cfg, 
                                        sess, 
                                        inputs, 
                                        outputs,
                                        outall=True) #getpose(image, cfg, sess, inputs, outputs, outall=False)

# scmap.shape   (22, 36, 22) --last dimension is joint
# locref       (22, 36, 22, 2)       
# pose.shape       (22, 3)            
# %%

# visualize.show_heatmaps(dlc_cfg, frame, scmap, pose)
# visualize.waitforbuttonpress()

plt_interp = "bilinear"
all_joints = dlc_cfg["all_joints"]
all_joints_names = dlc_cfg["all_joints_names"]
cmap="jet"
# subplot_width = 3
# subplot_height = math.ceil((len(all_joints) + 1) / subplot_width)
# f, axarr = plt.subplots(subplot_height, subplot_width)
for pidx, part in enumerate(all_joints):
    # plot_j = (pidx + 1) // subplot_width
    # plot_i = (pidx + 1) % subplot_width

    ## get heatmap for this bodypart
    scmap_part = np.sum(scmap[:, :, part], axis=2) #sum over last dim?
    # resize: cv2.resize fx and fy = 8 (bc  stride=8?) should it be dlc_cfg['stride']?
    scmap_part = imresize(scmap_part, 8.0, interpolationmethod=1) #---interp="bicubic"
    scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), "minimum") # (180, 292)

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    # curr_plot = axarr[plot_j, plot_i]
    plt.title('{} - {}'.format(pidx,all_joints_names[pidx]))
    # curr_plot.axis("off")
    plt.imshow(frame, interpolation=plt_interp)
    hm=plt.imshow(scmap_part, alpha=0.5, cmap=cmap, interpolation=plt_interp)
    plt.clim(vmin=0,vmax=1) 
    fig.colorbar(hm, ax=ax)
    
    plt.show()

plt.figure(figsize=(10,10))
plt.imshow(visualize.visualize_joints(frame, pose)) # (162, 288, 3)
plt.show()

# curr_plot = axarr[0, 0]
# curr_plot.set_title("Pose")
# curr_plot.axis("off")
# curr_plot.imshow(visualize_joints(img, pose))

#np.max(scmap[:, :, 15])

# %%
