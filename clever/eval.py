import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
from pgd.pgd_trainer import train, test
from logger import Logger
import utils
import numpy as np
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./pgd/cnfg.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data
    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'],

                                 )
    # initialization
    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model']).to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    model.load_state_dict(checkpoint['model'])
    model.float()

    test(cnfg, logger, 0, model, tst_loader, device)


def test(cnfg, logger, epoch, model, tst_loader, device):
    model.eval()
    acc, adv_acc = 0, 0
    for _, (x, y_) in enumerate(tqdm(tst_loader)):
        x,  y_ = x.to(device), y_.to(device)
        out = model(x)
        acc += (out.max(1)[1] == y_).sum().item() / len(y_)
        x_pgd = projected_gradient_descent(
            model, x,
            cnfg['pgd']['epsilon']/255,
            cnfg['pgd']['alpha']/255,
            7, np.inf)
        adv_out = model(x_pgd)
        adv_acc += (adv_out.max(1)[1] == y_).sum().item() / len(y_)
    logger.log_test(epoch, 0,
                    (acc/len(tst_loader))*100, "clean_testing")
    logger.log_test_adversarial(epoch, 0,
                                (adv_acc/len(tst_loader))*100, "pgd_testing")


if __name__ == "__main__":
    main()
