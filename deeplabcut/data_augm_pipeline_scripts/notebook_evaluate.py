import deeplabcut

config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'
NUM_SHUFFLES = 3
TRAINING_SET_INDEX = 0
GPU_TO_USE = 3
MODEL_PREFIX = ''

for sh in range(3):
    deeplabcut.evaluate_network(config_path, # config.yaml, common to all models
                                Shuffles=[sh],
                                trainingsetindex=TRAINING_SET_INDEX,
                                gputouse=GPU_TO_USE,
                                modelprefix=MODEL_PREFIX)