import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
from pgd.pgd_trainer import train, test
from logger import Logger
import utils
import time
from eval.spsa import spsa
import pgd.attack as pgd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eval/cnfg.yml', type=str)
    return parser.parse_args()


def main():
    start = time.time()
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data

    # initialization
    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])

    model = utils.get_model(cnfg['model']).to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    model.load_state_dict(checkpoint['model'])
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

    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 1)
    # test
    adv_acc = 0
    l, u = pgd.get_limits(device)
    for _, (inpt, targets) in enumerate(tst_loader):
        inpt, targets = inpt.to(device), targets.to(device)
        adv = spsa(model_fn=model, x=inpt, eps=8.0/255, clip_min=l, clip_max=u,
                   y=targets, nb_iter=10)
        # adv = spsa(model_fn=model, x=inpt, eps=8/255,
        #            y=targets, nb_iter=10)
        with torch.no_grad():
            adv_outpt = model(adv)
            adv_acc += (adv_outpt.max(1)[1] ==
                        targets).sum().item() / len(targets)
            # print(adv_acc)

    print('Total adv acc \t', adv_acc/len(tst_loader))
    # print('Test time: \t {}'.format(end-start))


if __name__ == "__main__":
    main()
