import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import yaml
import models


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def parse_config(path):
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


def get_model(cnfg):
    if cnfg['custom'] is True:
        raise NotImplementedError(
            "[ERROR] Custom models are currently not implemented")
    else:
        return models.get_tvision(cnfg['tvision']['name'], cnfg['tvision']['args'])


def get_scheduler(opt, cnfg, steps):
    if cnfg['lr_scheduler'] == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(opt,
                                                 base_lr=cnfg['lr_min'],
                                                 max_lr=cnfg['lr_max'],
                                                 step_size_up=steps/2,
                                                 step_size_down=steps/2)
    else:
        raise NotImplementedError(
            "[ERROR] The selected scheduler is not implemented")


def save_model(model, cnf, epoch, path):
    state = {
        'epoch': epoch,
        'cnf': cnf,
        'arch': type(model).__name__,
        'model': model.state_dict()
    }
    torch.save(state, path)
