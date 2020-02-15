import argparse
import logging
import os
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from original.utils import (upper_limit, lower_limit, std, clamp, get_loaders,
                            evaluate_pgd, evaluate_standard)
from tqdm import tqdm
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./mixed/cnfg.yml', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cnfg = utils.parse_config(args.config)

    utils.set_seed(cnfg['seed'])

    train_loader, test_loader = get_loaders(
        cnfg['data']['dir'], cnfg['data']['batch_size'])

    epsilon = (cnfg['pgd']['epsilon'] / 255.) / std
    alpha = (cnfg['pgd']['alpha'] / 255.) / std

    model = utils.get_model(cnfg['model']).cuda()
    checkpoint = torch.load(cnfg['resume']['path'])
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.float()

    logger = Logger(cnfg)

    opt = torch.optim.SGD(model.parameters(), lr=cnfg['train']['lr_max'],
                          momentum=cnfg['train']['momentum'], weight_decay=cnfg['train']['weight_decay'])
    amp_args = dict(opt_level=cnfg['opt']['level'],
                    loss_scale=cnfg['opt']['loss_scale'], verbosity=False)
    if cnfg['opt']['level'] == 'O2':
        amp_args['master_weights'] = cnfg['opt']['store']
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = cnfg['train']['epochs'] * len(train_loader)
    if cnfg['train']['lr_scheduler'] == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=cnfg['train']['lr_min'], max_lr=cnfg['train']['lr_max'],
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif cnfg['train']['lr_scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Evaluation
    model.eval()

    pgd_loss, pgd_acc = evaluate_pgd(
        test_loader, model, cnfg['pgd']['iter'], cnfg['pgd']['restarts'])
    test_loss, test_acc = evaluate_standard(test_loader, model)
    print('PGD \t {}, \t {}'.format(pgd_loss, pgd_acc))
    print('Natural \t {}, \t {}'.format(test_loss, test_acc))


if __name__ == "__main__":
    main()
