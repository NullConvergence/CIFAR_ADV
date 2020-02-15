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
    tr_loader, tst_loader = get_datasets(cnfg['data']['flag'],
                                         cnfg['data']['dir'],
                                         cnfg['data']['batch_size'],

                                         )
    # initialization
    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model']).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])

    scheduler = utils.get_scheduler(
        opt, cnfg['train'], cnfg['train']['epochs']*len(tr_loader))
    # train+test
    for epoch in range(cnfg['train']['epochs']):
        model.train()
        acc, tloss = 0, 0
        for batch_idx, (x, y_) in enumerate(tqdm(tr_loader)):
            x, y_ = x.to(device), y_.to(device)
            get adversarial from cleverhans
            x = projected_gradient_descent(
                model, x, cnfg['pgd']['epsilon']/255,
                cnfg['pgd']['alpha']/255,
                cnfg['pgd']['iter'],
                np.inf
            )
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y_)
            loss.backward()
            opt.step()
            acc += (output.max(1)[1] == y_).sum().item() / len(y_)
            tloss += loss
            scheduler.step()

        logger.log_train(epoch, tloss/len(tr_loader),
                         (acc/len(tr_loader))*100)
        # testing
        if (epoch+1) % cnfg['test'] == 0 or epoch == 0:
            test(cnfg, logger, epoch, model, tst_loader, device)
        # save
        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)


def test(cnfg, logger, epoch, model, tst_loader, device):
    model.eval()
    acc, adv_acc = 0, 0
    for _, (x, y_) in enumerate(tqdm(tst_loader)):
        x,  y_ = x.to(device), y_.to(device)
        out = model(x)
        acc += (out.max(1)[1] == y_).sum().item() / len(y_)
        x_pgd = projected_gradient_descent(
            model, x,  0.3, 0.01, 7, np.inf)
        adv_out = model(x_pgd)
        adv_acc += (adv_out.max(1)[1] == y_).sum().item() / len(y_)
    logger.log_test(epoch, 0,
                    (acc/len(tst_loader))*100, "clean_testing")
    logger.log_test_adversarial(epoch, 0,
                                (adv_acc/len(tst_loader))*100, "pgd_testing")


if __name__ == "__main__":
    main()
