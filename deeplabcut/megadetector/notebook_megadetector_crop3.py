# %%
# Set python path
#set PYTHONPATH="$PYTHONPATH:$HOME/git/cameratraps:$HOME/git/ai4eutils:$HOME/git/yolov5"
import sys
sys.path.insert(0, "/home/vic/git") #change the python path to where your git repos are (need megadetector related)

# sys.path.append("/home/vic/git/ai4eutils")
# sys.path.append("/home/vic/git/yolov5")
# for p in sys.path:
#      print(p)

# %%
# IMPORTS 
import json
import argparse
import glob
import os
import sys
import time
import warnings
import humanfriendly
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
import statistics
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

from PIL import Image, ImageFile, ImageFont, ImageDraw
from PIL import Image
from tqdm import tqdm
from cameratraps.ct_utils import truncate_float
from dlclive import DLCLive, Processor
from numpy import asarray
from numpy import savetxt
dlc_proc = Processor()

# %%
# variables to set
input_json_path = "/home/vic/vic_data/dlclive4mega/output2.json" #change to your json file
quadraped_model = "/home/vic/vic_data/dlclive4mega/models/DLC_ma_superquadruped_resnet_50_iteration-0_shuffle-1"
human_model =  "/home/vic/vic_data/dlclive4mega/models/DLC_human_dancing_resnet_101_iteration-0_shuffle-1"
macaque_model = "/home/vic/vic_data/dlclive4mega/models/DLC_monkey_resnet_50_iteration-0_shuffle-1"
# download the model, and set the path to it

# %%
# Draw bounding box - don't worry about this for now? 

""" def draw_bboxs(detections_list, im):
    """
    detections_list: list of set includes bbox.
    im: image read by Pillow.
    """
    
    for detection in detections_list:
        x1, y1,w_box, h_box = detection["bbox"]
        ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box
        draw = ImageDraw.Draw(im)
        
        imageWidth=im.size[0]
        imageHeight= im.size[1]
        (left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
        
        draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=4, fill='Red') """

# %%
#a method that allows using dlc live, but with different models, saves poses, and plots the poses onto image
def dlclive_pose(model, crop_np, crop, fname, index):
    dlc_live = DLCLive(model, processor=dlc_proc)
    dlc_live.init_inference(crop_np)
    keypts = dlc_live.get_pose(crop_np)  #(20, 3): x, y, confidence
    savetxt(str(fname)+ '_' + str(index) + '.csv' , keypts, delimiter=',')
    xpose = []
    ypose = []
    for key in keypts[:,2]:
        if key < 0.05:
            index = np.where(keypts[:,2]==key)
            xpose.append(keypts[index,0])
            ypose.append(keypts[index,1])
    plt.imshow(crop)
    plt.scatter(xpose[:], ypose[:], 40, color='cyan')
    plt.savefig(str(fname)+ '_' + str(index) + '.png')
    plt.show()
    



# %%
###########################################
# Read detections from json
with open(input_json_path, 'r') as f:
    detection_results = json.load(f)





# %%
""" 
for every image, extract bounding box with high confidence
use different model for dlc live depending on if animal / human

 """
for img_data in detection_results["images"]:
    img = Image.open(img_data['file'])
    basename = os.path.basename(img_data['file'])
    fname = os.path.splitext(basename)[0]
    # dlc_live.init_inference(np.asarray(img)) #---why this needed? should it be outside the loop?
    # keypts = dlc_live.get_pose(np.asarray(img)) #(20, 3): x, y, confidence

    plt.imshow(img)
    # plt.scatter(keypts[:,0], keypts[:,1], 20,
    #             color='r')
    plt.show()
    # plt.imshow(img)
    for detections_dict in img_data["detections"]:
        index = img_data["detections"].index(detections_dict)
        if detections_dict["conf"] > 0.8: 
            x1, y1,w_box, h_box = detections_dict["bbox"]
            ymin,xmin,ymax, xmax = y1, x1, y1 + h_box, x1 + w_box
            
            imageWidth=img.size[0]
            imageHeight= img.size[1]
            area = (xmin * imageWidth, ymin * imageHeight, xmax * imageWidth,
                    ymax * imageHeight)
            crop = img.crop(area)
            plt.imshow(crop)
            plt.show()
            crop_np = np.asarray(crop)
            if detections_dict["category"] == "1":
                print("which model would like like to use? quadraped or macaque?")
                model = str(input())
                if model == 'quadraped':
                    dlclive_pose(quadraped_model, crop_np, crop, fname, index)
                elif model == 'macaque':
                    dlclive_pose(macaque_model, crop_np, crop, fname, index)
            elif detections_dict["category"] == "2":
                dlclive_pose(human_model, crop_np, crop, fname, index)


        
        



# %%
