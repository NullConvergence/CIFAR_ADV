import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
from pgd.pgd_trainer import test
from free.free_trainer import train
import clean.trainer as clean
from pgd.attack import get_eps_alph
from logger import Logger
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./free/cnfg.yml', type=str)
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
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model']).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])
    amp_args = dict(opt_level=cnfg['opt']['level'],
                    loss_scale=cnfg['opt']['loss_scale'], verbosity=False)
    if cnfg['opt']['level'] == '02':
        amp_args['master_weights'] = cnfg['opt']['store']
    model, opt = amp.initialize(model, opt, **amp_args)
    scheduler = utils.get_scheduler(
        opt, cnfg['train'], cnfg['train']['epochs']*len(tr_loader))

    # train+test
    delta = torch.zeros(cnfg['data']['batch_size'], 3, 32, 32).to(device)
    delta.requires_grad = True
    epsilon, _ = get_eps_alph(
        cnfg['pgd']['epsilon'], cnfg['pgd']['alpha'], device)
    epoch = 0
    while epoch <= cnfg['train']['epochs']:
        if epoch <= cnfg['mixed']['adv_epochs']:
            print('[INFO][TRAIN] \t Training with Adversarial Examples')
            delta = train(epoch, delta, cnfg['train']['batch_replay'],
                          epsilon, model, criterion, opt, scheduler,
                          tr_loader, device, logger)
            # always test with pgd
            if epoch % cnfg['test'] == 0 or epoch == 0:
                print(
                    '[INFO][TEST] \t Testing with both Adversarial and Clean Examples')
                test(epoch, model, tst_loader, criterion,
                     device, logger, cnfg, opt)
            epoch += cnfg['train']['batch_replay']
        else:
            print('[INFO][TRAIN] \t Training with Clean Examples')
            clean.train(epoch, model, criterion,
                        opt, scheduler, tr_loader, device, logger)
            if (epoch+1) % cnfg['test'] == 0 or epoch == 0:
                test(epoch, model, tst_loader, criterion,
                     device, logger, cnfg, opt)
            epoch += 1
        # save
        if epoch % cnfg['save']['epochs'] == 0 or (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)


if __name__ == "__main__":
    main()
