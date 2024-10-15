
import glob
import itertools
import os
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from utils import *

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment("PGRNet")
ex.captured_out_filter = apply_backspaces_and_linefeeds


###### Set up source folder ######
source_folders = ['.', './dataloaders', './models', './utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    """Default configurations"""
    seed = 2021
    gpu_id = 0
    num_workers = 0  # 0 for debugging.
    mode = 'train'

    ## dataset
    dataset = 'CHAOST2'  # i.e. abdominal MRI - 'CHAOST2'; cardiac MRI - CMR
    exclude_label = None  # None, for not excluding test labels;
    # 1 for Liver, 2 for RK, 3 for LK, 4 for Spleen in 'CHAOST2'
    if dataset == 'CMR':
        n_sv = 1000
    else:
        n_sv = 5000
    min_size = 200
    max_slices = 3
    use_gt = False  # True - use ground truth as training label, False - use supervoxel as training label
    eval_fold = 0  # (0-4) for 5-fold cross-validation
    test_label = [1, 4]  # for evaluation
    supp_idx = 0  # choose which case as the support set for evaluation, (0-4) for 'CHAOST2', (0-7) for 'CMR'
    n_part = 3  # for evaluation, i.e. 3 chunks

    ## training
    n_steps = 1000
    batch_size = 1
    n_shot = 1
    n_way = 1
    n_query = 1
    lr_step_gamma = 0.95
    bg_wt = 0.1
    t_loss_scaler = 0.0
    ignore_label = 255
    print_interval = 100
    # save_snapshot_every = 1000 # 已经有更好的保存方式了，这被弃用
    max_iters_per_load = 1000  # epoch size, interval for reloading the dataset
    alpha=0.9 # dual-scale
    beta=1.0
    
    # Network
    reload_model_path = None
    
    # Prototype Refinement
    n_iters=7

    # visualize
    visualize = False
    vis_path = 'vis/CHAOST2/1/'

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_str = '_'.join(
        [mode]
        + [dataset, ]
        + [f'cv{eval_fold}'])

    path = {
        'log_dir': './test_results',
        'CHAOST2': {'data_dir': '../../data/CHAOST2'},
        'SABS': {'data_dir': '../../data/SABS'},
        'CMR': {'data_dir': '../../data/CMR'},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    # observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    # ex.observers.append(observer)

    # observer_mongo = MongoObserver.create(os.path.join(url='localhost:27017', db_name="PGRNet")
    # ex.observers.append(observer_mongo)



    ex.observers.append(FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name)))
    # mongo_url = 'localhost:27017'
    # ex.observers.append(MongoObserver(url=mongo_url, db_name="PGRNet"))


    # if fileStorage:
    #     observer_file = FileStorageObserver(config["log_dir"])
    #     ex.observers.append(observer_file)
    #
    # if MongoDB:
    #     try:
    #         host, port = config["mongo_host"], config["mongo_port"]
    #         observer_mongo = MongoObserver(url=f"{host}:{port}", db_name=db_name)
    #         ex.observers.append(observer_mongo)
    #     except ModuleNotFoundError:
    #         # Ignore Mongo Observer
    #         pass


    return config
