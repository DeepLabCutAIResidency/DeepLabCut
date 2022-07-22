# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
import os
import cv2
from imgaug import augmenters as iaa
import imgaug as ia


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def polygonArea(X, Y): 
    # X and Y are numpy arrays of size nrows=N, ncolumns=2

    n = X.shape[0]
    # Initialize area
    area = 0.0
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
    # Return absolute value
    return int(abs(area / 2.0))

################################################
## Load batch data



#######################################
## Define augmentation sequence
ia.seed(1)

# example from imgaug
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                 iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8))
], 
random_order=True) # apply augmenters in random order

###############################
# Apply to batch of images
images_aug = seq(images=images)

##########################################
# %%
from deeplabcut.pose_estimation_tensorflow.datasets import Batch, PoseDatasetFactory
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config

# %%
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'#'/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
SHUFFLE_ID=0
TRAINING_SET_IDX=0
MODEL_PREFIX=''

cfg = read_config(config_path)
train_cfg_path,\
_,_ = deeplabcut.return_train_network_path(config_path,
                                            shuffle=SHUFFLE_ID,
                                            trainingsetindex=TRAINING_SET_IDX, 
                                            modelprefix=MODEL_PREFIX)
train_cfg = read_config(train_cfg_path)

# %%
dataset = PoseDatasetFactory.create(train_cfg)


# Next batch
# batch = dataset.next_batch()

(batch_images,
 joint_ids,
 batch_joints,
 data_items,
 sm_size,
 target_size) = dataset.get_batch()

pipeline = dataset.build_augmentation_pipeline(height=target_size[0], 
                                               width=target_size[1], 
                                               apply_prob=0.5)
batch_images, batch_joints = pipeline(images=batch_images, 
                                      keypoints=batch_joints)


# If you would like to check the augmented images, script for saving
# the images with joints on:
# import imageio
# for i in range(self.batch_size):
#    joints = batch_joints[i]
#    kps = KeypointsOnImage([Keypoint(x=joint[0], y=joint[1]) for joint in joints], shape=batch_images[i].shape)
#    im = kps.draw_on_image(batch_images[i])
#    imageio.imwrite('some_location/augmented/'+str(i)+'.png', im)
# %%
