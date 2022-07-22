## Initialise dict with additional edits to train config: optimizer
train_edits_dict = {}
dict_optimizer = {'optimizer':'adam',
    'batch_size': 8,
    'multi_step': [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]} # if no yaml file passed, initialise as an empty dict
train_edits_dict.update({'optimizer': dict_optimizer['optimizer'], #'adam',
    'batch_size': dict_optimizer['batch_size'], #16,
    'multi_step': dict_optimizer['multi_step']}) # learning rate schedule for adam: [[1e-4, 7500], [5 * 1e-5, 12000], [1e-5, 200000]]