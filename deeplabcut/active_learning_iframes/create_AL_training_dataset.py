import os, sys
import re 
import argparse
import yaml
import deeplabcut
import pickle
import pandas as pd

from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset



###################################
# Inputs
config_path = '/home/sofia/datasets/Horse10_AL/Horses-Byron-2019-05-08/config.yaml' 
pose_cfg_yaml_adam_path = '/home/sofia/DeepLabCut/deeplabcut/adam_pose_cfg.yaml'

NUM_SHUFFLES = 3
create_training_dataset(config_path,
                        num_shuffles=NUM_SHUFFLES,
                        userfeedback=True,
                        net_type='resnet_50',
                        trainIndices=list_train_idcs_per_shuffle,
                        testIndices=list_test_idcs_per_shuffle,
                        posecfg_template=pose_cfg_yaml_adam_path,
                        ) # augmenter_type=None, posecfg_template=None,

    # Shuffles=None,
    # windows2linux=False,
    # userfeedback=False,
    # trainIndices=None,
    # testIndices=None,
    # net_type=None,
    # augmenter_type=None,
    # posecfg_template=None,