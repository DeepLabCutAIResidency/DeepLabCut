"""
Extract i-frames from Horse dataset

"""

#####################################################
# %%
import deeplabcut
import os
import pandas as pd
import cv2
import subprocess

#################################################
def extract_i_frames(video_path):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_path]).decode()
    f_types = out.replace('pict_type=','').split()
    return [i for i, type_ in enumerate(f_types) if type_ == 'I']
###################################################
# %% 
# Input params
project_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08'
labelled_data_h5file = \
    os.path.join(project_dir,'training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5') # h5 file

video_FPS=30

video_temp_path = 'temp.avi'

#################################################
# %% 
# Read dataframe with labelled data
df = pd.read_hdf(labelled_data_h5file)

list_paths_to_files = [os.path.join(*el) for el in df.index]
# sort in original order!
l=[os.path.join(v) for u,v,w in df.index]
list_subdirs = sorted(set(l),
                      key=l.index)

map_subdirs_to_files = dict()
for d in list_subdirs:
    map_subdirs_to_files[d]  = [el for el in list_paths_to_files if d in el]
    
# labeled_data_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08/labeled-data'
# list_subdirs = os.listdir(labeled_data_dir)
# for dir in list_subdirs:
#     print(dir)
##############################################################
# %% Extract i-frames per video

for d in map_subdirs_to_files.keys():

    # get image size of all files in video
    list_h_w_img = \
        [cv2.imread(f).shape for f in map_subdirs_to_files[d]]
    list_h_w_img =  [(w,h) for h,w,c in list_h_w_img]
    if not all([list_h_w_img[0]==l for l in list_h_w_img]):
        break
    img_size = list_h_w_img[0]

    # initialise video for this dir---use context?
    out = cv2.VideoWriter(video_temp_path, #'temp.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          video_FPS, img_size)

    # write frames to video
    for f in map_subdirs_to_files[d]:
        img = cv2.imread(os.path.join(project_dir,f)) #-----------check path
        out.write(img)      
    out.release()

    # extract i-frames
    list_idx_i_frames_wrt_video = extract_i_frames(video_temp_path)