"""
To run mirror training:
nohup python deeplabcut/data_augm_pipeline_scripts/data_augm_train_all_models.py /media/data/stinkbugs-DLC-2022-07-15_MIRROR/config.yaml 'data_augm' 0  --train_iteration=1 > log_gpu_0_mirror.txt &

"""         

import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import edit_config
import numpy as np
import os, shutil, sys

##########################################
# always SHUFFLE=1
bug_body_len_estim = 220.3 # px #np.sqrt(area_bug)

start_fraction = 0
end_fraction = 2
n_models = 5

base_project_path = '/media/data/stinkbugs-DLC-2022-07-15_COVERING'

##########################################
np_fractions = np.linspace(start_fraction, 
                           end_fraction,
                           num = n_models)
for i in np_fractions:
    
    ## Copy base project 
    shutil.copytree(base_project_path, 
                    base_project_path+'_'+str(int(i*100)))

    ## Get train cfg path of copy dir
    train_cfg_path = \
    '/media/data/stinkbugs-DLC-2022-07-15_COVERING_{}/dlc-models/iteration-1/stinkbugsJul15-trainset80shuffle0/train/pose_cfg.yaml'.format(int(i*100))

    ## Edit
    if i==0:
        edit_config(train_cfg_path,
                    {'covering': False, 
                    'convolution': {'emboss': False,
                                    'sharpen': False}})
    else:
        print(float(round(1/(i*bug_body_len_estim),3)))
        edit_config(train_cfg_path,
                    {'covering': True, 
                    'covering_size_percent': float(round(1/(i*bug_body_len_estim),3)),
                    'convolution': {'emboss': False,
                                    'sharpen': False}})

                