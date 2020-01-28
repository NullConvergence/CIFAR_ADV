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
    tr_loader, tst_loader = get_datasets(cnfg['data']['flag'],
                                         cnfg['data']['dir'],
                                         cnfg['data']['batch_size'])
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
    preproc = dict(mean=mean, std=std)

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10,
                                         preprocessing=preproc)
    attack = foolbox.attacks.FGSM(fmodel)

    for batch_idx, (x, y_) in enumerate(tst_loader):
        x_np, y_np_ = x.numpy(), y_.numpy()

        adversarials = attack(x_np, y_np_)
        print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == y_np_))


if __name__ == '__main__':
    main()
