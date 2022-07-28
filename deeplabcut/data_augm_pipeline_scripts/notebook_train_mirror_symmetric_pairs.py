import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import edit_config

SHUFFLE_ID=0
train_cfg_path = \
'/media/data/stinkbugs-DLC-2022-07-15_MIRROR/data_augm_02_mirror_symmetric/dlc-models/iteration-1/stinkbugsJul15-trainset80shuffle'+\
    str(SHUFFLE_ID)+\
    '/train/pose_cfg.yaml' # should be original copy of the same shuffle in data_augm_01_mirror


# train_cfg = read_config(train_cfg_path)

edit_config(train_cfg_path,
            {'fliplr': True, 
            'mirror': False, #---- ATT mirror should be set  to false!
            'symmetric_pairs': [(0,12), 
                                  (1,13),
                                  (2,14),
                                  (3,15),
                                  (4,16),
                                  (5,17),
                                  (6,18),
                                  (7,19),
                                  (8,20),
                                  (9,21),
                                  (10,22),
                                  (11,23),
                                  (24,25),
                                  (27,31),
                                  (28,32),
                                  (29,33),
                                  (30,34),
                                  (35,36),
                                  (39,40)]})

"""
To run mirror training:
nohup python deeplabcut/data_augm_pipeline_scripts/data_augm_train_all_models.py /media/data/stinkbugs-DLC-2022-07-15_MIRROR/config.yaml 'data_augm' 0  --train_iteration=1 > log_gpu_0_mirror.txt &



"""                        