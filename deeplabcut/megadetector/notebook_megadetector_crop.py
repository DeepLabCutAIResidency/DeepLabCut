# %%
# Set python path
#set PYTHONPATH="$PYTHONPATH:$HOME/git/cameratraps:$HOME/git/ai4eutils:$HOME/git/yolov5"
import sys
# sys.path.append("/home/vic/git/ai4eutils")
sys.path.insert(0, "/home/vic/git")
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
import pandas as pd
from PIL import Image, ImageFile, ImageFont, ImageDraw
import statistics
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from tqdm import tqdm

from cameratraps.ct_utils import truncate_float

# %%
input_json_path = "/home/vic/vic_data/dlclive4mega/output.json"

# %%
# Draw bounding box 
def draw_bboxs(detections_list, im):
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
               (right, top), (left, top)], width=4, fill='Red')

size = (480,270)
im = Image.open("../input/iwildcam2021-fgvc8/test/8d11eea8-21bc-11ea-a13a-137349068a90.jpg")
im = im.resize(size)

# Overwrite bbox
draw_bboxs(detection_results[0]['detections'], im)

# Show
plt.imshow(im)
plt.title(f"image with bbox")

# %%
###########################################
# Read detections from json
with open(input_json_path, 'r') as f:
    detection_results = json.load(f)
# %%
# for every image, for every bounding box, extract the bounding box

for img_data in detection_results["images"]:
    img = Image.open(img_data['file'])
    plt.imshow(img)
    for detections_dict in img_data["detections"]:
        x1, y1,w_box, h_box = detections_dict["bbox"]
        ymin,xmin,ymax, xmax = y1, x1, y1 + h_box, x1 + w_box
        
        imageWidth=img.size[0]
        imageHeight= img.size[1]
        area = (xmin * imageWidth, ymin * imageHeight, xmax * imageWidth,
                ymax * imageHeight)
        out = img.crop(area)

        plt.imshow(out)
        plt.show()



# %%
