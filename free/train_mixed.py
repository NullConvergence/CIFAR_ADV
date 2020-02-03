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
    # change learnign rate scheduler and nr of epochs
    if cnfg['train']['milestones']:
        cnfg['train']['milestones'] = [x//cnfg['train']['epochs']
                                       for x in cnfg['train']['milestones']]
    cnfg['train']['epochs'] = cnfg['train']['epochs'] // \
        cnfg['train']['batch_replay']
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
    every_n = cnfg['mixed']['every_n']
    buff, adv_buff = 0, False
    for epoch in range(cnfg['train']['epochs']):
        if every_n is True:
            # we start by training with adversarial examples
            # for a clean training start, write 'epoch +1'
            print(epoch % cnfg['mixed']['n'])
            if epoch % cnfg['mixed']['n'] == 0 and adv_buff is False:
                # start adv training
                adv_buff = True
                buff = 0
            elif adv_buff is True and buff == cnfg['mixed']['adv_epochs']:
                # reset buffers
                adv_buff, buff = False, 0

            if adv_buff is True and buff < cnfg['mixed']['adv_epochs']:
                # train advesarial
                print('[INFO][TRAIN] \t Training with Adversarial Examples')
                delta = train(epoch, delta, cnfg['train']['batch_replay'],
                              epsilon, model, criterion, opt, scheduler,
                              tr_loader, device, logger, cnfg['train']['lr_scheduler'])
                buff += 1
            else:
                print('[INFO][TRAIN] \t Training with Clean Examples')
                clean.train(epoch, model, criterion,
                            opt, scheduler, tr_loader, device, logger,
                            cnfg['train']['lr_scheduler'])
        else:
            if epoch < cnfg['mixed']['adv_epochs']:
                print('[INFO][TRAIN] \t Training with Adversarial Examples')
                delta = train(epoch, delta, cnfg['train']['batch_replay'],
                              epsilon, model, criterion, opt, scheduler,
                              tr_loader, device, logger, cnfg['train']['lr_scheduler'])
            else:
                print('[INFO][TRAIN] \t Training with Clean Examples')
                clean.train(epoch, model, criterion,
                            opt, scheduler, tr_loader, device, logger,
                            cnfg['train']['lr_scheduler'])
        # always test with pgd
        print('[INFO][TEST] \t Testing with both Adversarial and Clean Examples')
        test(epoch, model, tst_loader, criterion,
             device, logger, cnfg, opt)
        # save
        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)


if __name__ == "__main__":
    main()
