# %%
import deeplabcut
import os
import pandas as pd
import cv2
import subprocess

import matplotlib.pyplot as plt
import numpy as np
#################################################
# %% 
def extract_i_frames(video_path):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_path]).decode()
    f_types = out.replace('pict_type=','').split()
    return [i for i, type_ in enumerate(f_types) if type_ == 'I']

######################################################
# %%
path_to_video = '/media/data/stinkbugs/videos/cam1_04-Mar-21_20-27-56.mov'
# '/media/data/crayfish/otherVideos/2crayfish.mp4'


idcs_i_frames_wrt_video = extract_i_frames(path_to_video)

print(np.diff(np.asarray(idcs_i_frames_wrt_video))) 
plt.plot(np.diff(np.asarray(idcs_i_frames_wrt_video)),'.')
plt.xlabel('frame')
plt.ylabel('step between consecute keyframes')
# %%
