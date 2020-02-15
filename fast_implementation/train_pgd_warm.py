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


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch-size', default=128, type=int)
#     parser.add_argument('--data-dir', default='../../cifar-data', type=str)
#     parser.add_argument('--epochs', default=1, type=int)
#     parser.add_argument('--lr-schedule', default='multistep',
#                         type=str, choices=['cyclic', 'multistep'])
#     parser.add_argument('--lr-min', default=0., type=float)
#     parser.add_argument('--lr-max', default=0.2, type=float)
#     parser.add_argument('--weight-decay', default=5e-4, type=float)
#     parser.add_argument('--momentum', default=0.9, type=float)
#     parser.add_argument('--epsilon', default=8, type=int)
#     parser.add_argument('--attack-iters', default=7,
#                         type=int, help='Attack iterations')
#     parser.add_argument('--restarts', default=1, type=int)
#     parser.add_argument('--alpha', default=2, type=int, help='Step size')
#     parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
#                         help='Perturbation initialization method')
#     parser.add_argument('--out-dir', default='train_pgd_output',
#                         type=str, help='Output directory')
#     parser.add_argument('--seed', default=0, type=int, help='Random seed')
#     parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
#                         help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
#     parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
#                         help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
#     parser.add_argument('--master-weights', action='store_true',
#                         help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
#     return parser.parse_args()


def main():
    args = parse_args()
    cnfg = utils.parse_config(args.config)

    np.random.seed(cnfg['seed'])
    torch.manual_seed(cnfg['seed'])
    torch.cuda.manual_seed(cnfg['seed'])

    train_loader, test_loader = get_loaders(
        cnfg['data']['dir'], cnfg['data']['batch_size'])

    epsilon = (cnfg['pgd']['epsilon'] / 255.) / std
    alpha = (cnfg['pgd']['alpha'] / 255.) / std

    model = utils.get_model(cnfg['model']).cuda()

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

    # Training
    print('starting epoch')
    for epoch in range(cnfg['train']['epochs']):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.cuda(), y.cuda()
            if epoch <= cnfg['mixed']['adv_epochs']:
                delta = torch.zeros_like(X).cuda()
                if cnfg['pgd']['delta-init'] == 'random':
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i]
                                                   [0][0].item(), epsilon[0][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                for _ in range(cnfg['pgd']['iter']):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(
                        delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
                delta = delta.detach()
                output = model(X + delta)
                loss = criterion(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            else:
                output = model(X)
                loss = criterion(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.log_train(epoch, train_loss/train_n, train_acc/train_n)
        # test normal
        test_loss, test_acc = evaluate_standard(test_loader, model)
        logger.log_test(epoch, test_loss, test_acc)
        # test adv
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 7, 1)
        logger.log_test_adversarial(
            epoch,  pgd_loss/len(test_loader), pgd_acc/len(test_loader))

        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)

    # Evaluation
    # model_test = PreActResNet18().cuda()
    # model_test.load_state_dict(model.state_dict())
    # model_test.float()
    # model_test.eval()

    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 7, 1)
    # test_loss, test_acc = evaluate_standard(test_loader, model_test)

    # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f',
    #             test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
