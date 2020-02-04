import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
import pgd.pgd_trainer as pgd
import clean.trainer as clean
from logger import Logger
import utils
import foolbox
import numpy as np

import foolbox.ext.native as fbn


# TODO: import from cifar_data
mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eval/cnfg.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data
    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'],
                                 apply_transform=False)
    # initialization
    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    # logger = Logger(cnfg)
    model = utils.get_model(cnfg['model'])

    checkpoint = torch.load(cnfg['resume']['path'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    preproc = dict(mean=mean, std=std, axis=-3)

    fmodel = fbn.models.PyTorchModel(
        model, bounds=(0, 1), preprocessing=preproc)

    attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
    pgd = fbn.attacks.ProjectedGradientDescentAttack(fmodel)

    acc = 0

    for _, (x, y_) in enumerate(tst_loader):
        x, y_ = x.cuda(), y_.cuda()

        adversarials = pgd(x, y_, epsilon=0.8/255.,
                           step_size=0.2/255.,
                           num_steps=40)
        tmp = fbn.utils.accuracy(fmodel, adversarials, y_)
        acc += tmp

    print('Final acc \t {}'.format(acc / len(tst_loader)))


if __name__ == '__main__':
    main()
