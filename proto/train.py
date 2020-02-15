import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
from proto.trainer import train, test
from logger import Logger
from gensim.models import KeyedVectors
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./proto/cnfg.yml', type=str)
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
    word2vec = KeyedVectors.load_word2vec_format(
        cnfg['word2vecpath'], binary=True)
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
    for epoch in range(cnfg['train']['epochs']):
        train(epoch, model, word2vec,
              opt, scheduler, tr_loader, device, logger)
        # testing
        if (epoch+1) % cnfg['test'] == 0 or epoch == 0:
            test(epoch, model, word2vec, tst_loader, device, logger)
        # save
        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)


if __name__ == "__main__":
    main()
