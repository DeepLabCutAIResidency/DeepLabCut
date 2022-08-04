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

import matplotlib.pyplot as plt
import numpy as np
#################################################
# %% 
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

video_FPS=60
step_frames_unif = 25

video_temp_path = '/home/sofia/datasets/temp.avi'

#################################################
# %% 
# Read dataframe with labelled data
df = pd.read_hdf(labelled_data_h5file)
# ----> ensure frames per video are sorted? assuming name corresponds to frame number

list_paths_to_files = [os.path.join(project_dir,*el) for el in df.index]
# when making list with unique elements, ensure they are sorted in the same \
# order as in original!
l=[os.path.join(v) for u,v,w in df.index]
list_subdirs = sorted(set(l),key=l.index)

# map subdirectory to files inside it ---maybe specify extension?
map_subdirs_to_files = dict()
for d in list_subdirs:
    map_subdirs_to_files[d]  = [os.path.join(project_dir,u,v,w) for u,v,w in df.index if d==v] # ---improve this bit
    
# labeled_data_dir = '/home/sofia/datasets/Horses-Byron-2019-05-08/labeled-data'
# list_subdirs = os.listdir(labeled_data_dir)
# for dir in list_subdirs:
#     print(dir)
##############################################################
# %% Extract i-frames per video
# loop thru directories (one directory=one video)
bool_iframes_in_df = [False]*len(df)
list_step_btw_iframes = []
map_subdirs_to_iframes_fraction = dict()
for d in list(map_subdirs_to_files.keys()):

    # get image size (assuming common to all files in this directory)
    list_wh_img = [tuple([cv2.imread(f).shape[i] for i in [1,0]]) for f in map_subdirs_to_files[d]]
    if not all([list_wh_img[0]==l for l in list_wh_img]):
        break
    img_size_wh = list_wh_img[0]

    # initialise video for this dir---use context?
    out = cv2.VideoWriter(video_temp_path, #'temp.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          video_FPS, 
                          img_size_wh)

    # write frames to video
    for f in map_subdirs_to_files[d]:
        img = cv2.imread(f) 
        out.write(img)      
    out.release()

    # extract i-frames (idcs relative to video)
    idcs_i_frames_wrt_video = extract_i_frames(video_temp_path)

    list_step_btw_iframes.append(np.diff(np.asarray(idcs_i_frames_wrt_video))) # print step between iframes
    map_subdirs_to_iframes_fraction[d] = (len(idcs_i_frames_wrt_video),
                                          len(map_subdirs_to_files[d]),
                                          len(idcs_i_frames_wrt_video)/len(map_subdirs_to_files[d]))
    print('I-frames in {}: {}/{} = {}'.format(d,
                                            len(idcs_i_frames_wrt_video),
                                            len(map_subdirs_to_files[d]),
                                            len(idcs_i_frames_wrt_video)/len(map_subdirs_to_files[d])))# print fraction of iframes
    
    # inspect how many frames from video are iframes
    # plt_bool_list_iframes = [True if j in idcs_i_frames_wrt_video else False for j in range(len(map_subdirs_to_files[d]))]
    # plt_bool_list_unif = [True if j%step_frames_unif==0 else False for j in range(len(map_subdirs_to_files[d]))]
    # # l=[i for i,x in enumerate(bool_list) if x]; l==idcs_i_frames_wrt_video --- True
    # plt.plot(plt_bool_list_iframes,'r.',label='iframes')
    # plt.plot(plt_bool_list_unif,'b.',label='unif')
    # plt.xlim([0,50])
    # plt.xlabel('frame')
    # plt.ylabel('i-frame == True')
    # plt.show()

    ## get frames filenames for i-frames
    # list_frames_png_str = map_subdirs_to_files[d]
    list_i_frames_png_str = [map_subdirs_to_files[d][j] for j in idcs_i_frames_wrt_video]

    # add to boolean vector for dataframe
    bool_iframes_in_df = [x or y for x,y in zip(bool_iframes_in_df,
                                                [True if el in list_i_frames_png_str else False 
                                                    for el in list_paths_to_files])] 
    # bool_iframes_in_df = bool_iframes_in_df or \
    #                      [True if el in list_i_frames_png_str else False \
    #                       for el in list_paths_to_files] 
    # check:
    # idcs_where_true = [i for i,x in enumerate(bool_iframes_in_df) if x]
    # [list_paths_to_files[j] for j in idcs_where_true]==list_i_frames_png_str

######################################################
# %%
# Plot fraction of i-frames per video
plt.plot(map_subdirs_to_iframes_fraction.keys(),
        [w*100 for u,v,w in map_subdirs_to_iframes_fraction.values()],'.')
plt.xticks(range(len(map_subdirs_to_iframes_fraction)), 
            list(map_subdirs_to_iframes_fraction.keys()),
            rotation = 45, ha="right", fontsize='x-small')

plt.ylabel('fraction of i-frames (%)')
plt.ylim([5,10])
plt.show()
####################################################
# %%
# add results to dataframe and 
df.insert(0,"I-frame",bool_iframes_in_df)

df.to_hdf(labelled_data_h5file.split('.h5')[0]+'_iframes.h5',
          "df_with_missing",
          format="table",
          mode="w")
#################################################
# %% delete temp video if required (eventually user input whether to keep it or not?)
