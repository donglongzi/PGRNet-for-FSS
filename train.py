import os
import random
import logging
import shutil

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import resnet101
from dataloaders.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex
# import fitlog
import socket
import wandb

from utils import Scores
import warnings
warnings.filterwarnings("ignore")

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

@ex.automain
def main(_run, _config, _log):

    wandb.init(config=_config,
               project='PGRNet',
               entity="dragon-group",
               notes=socket.gethostname(),
               name=_config['path']['log_dir'],
               dir = './',
               job_type = "training",
                     reinit = True)

    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = resnet101()

    # load pretrained model
    model = load_resnet_param(model, stop_layer='layer4', layer_num=101)
    model = nn.DataParallel(model, [0])

    # disable the  gradients of not optomized layers
    turn_off(model)

    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * (_config['max_iters_per_load']//_config['batch_size']) for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'proto_loss':0}

    i_iter = 0
    _log.info(f'Start training...')

    best_dice = 0
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        scores = Scores()
        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            sample['support_images'] = sample['support_images'].permute(1, 2, 0, 3, 4, 5)
            sample['support_fg_labels'] = sample['support_fg_labels'].permute(1, 2, 0, 3, 4)
            sample['query_images'] = sample['query_images'].permute(1, 0, 2, 3, 4)

            support_images = torch.cat([torch.cat([shot.float().cuda() for shot in way], dim=0)
                              for way in sample['support_images']], dim=0)
            support_fg_mask = torch.cat([torch.cat([shot.float().cuda() for shot in way], dim=0)
                               for way in sample['support_fg_labels']], dim=0)

            query_images = torch.cat([query_image.float().cuda() for query_image in sample['query_images']], dim=0)
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            # Compute outputs and losses.
            query_pred, aux_mask, proto_loss = model(query_images, support_images, support_fg_mask.unsqueeze(dim=1))
            query_pred = nn.functional.interpolate(query_pred, size=query_labels.shape[-2:], mode='bilinear', align_corners=True)

            query_pred = nn.functional.softmax(query_pred, dim=1)
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)
            loss = query_loss + _config['alpha'] * proto_loss

            query_pred = query_pred.argmax(dim=1) # C x H x W

            # Record scores.
            scores.record(query_pred, query_labels)

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            proto_loss = proto_loss.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)
            _run.log_scalar('proto_loss', proto_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['proto_loss'] += proto_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                proto_loss = log_loss['proto_loss'] / _config['print_interval']

                wandb.log({'total_loss': total_loss}, step=i_iter + 1)
                wandb.log({'query_loss': query_loss}, step=i_iter + 1)

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['proto_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},'
                          f'proto_loss: {proto_loss}' f' dice score: {scores.patient_dice[-1].item()}')

            if best_dice < scores.patient_dice[-1].item():
                best_dice = scores.patient_dice[-1].item()
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', 'best_model.pth'))
            i_iter += 1

    # saving final checkpoint
    _log.info('###### saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', 'final_model.pth'))

    _log.info('End of training.')
    wandb.finish()
    return 1

