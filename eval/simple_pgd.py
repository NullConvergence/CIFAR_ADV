import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
from logger import Logger
import utils
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eval/cnfg.yml', type=str)
    return parser.parse_args()


def pgd_attack(model, images, labels, device, eps=8/255, alpha=2/255, iters=40):
    loss = nn.CrossEntropyLoss()
    ori_images = images.data
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def main():
    start = time.time()
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

    model = utils.get_model(cnfg['model']).to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    model.load_state_dict(checkpoint['model'])
    model.float()
    model.eval()

    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])
    amp_args = dict(opt_level=cnfg['opt']['level'],
                    loss_scale=cnfg['opt']['loss_scale'], verbosity=False)
    if cnfg['opt']['level'] == '02':
        amp_args['master_weights'] = cnfg['opt']['store']
    model, opt = amp.initialize(model, opt, **amp_args)
    # test
    acc, adv_acc = 0, 0
    for batch_idx, (x, y_) in enumerate(tst_loader):
        x, y_ = x.to(device), y_.to(device)
        adv = pgd_attack(model, x, y_, device, eps=cnfg['pgd']['epsilon']/255,
                         alpha=cnfg['pgd']['alpha']/255,
                         iters=cnfg['pgd']['iter'])
        out = model(x)
        adv_out = model(adv)

        acc += (out.max(1)[1] == y_).sum().item() / len(y_)
        adv_acc += (adv_out.max(1)[1] == y_).sum().item() / len(y_)

    print('Natural Accuracy: \t {}'.format(acc/len(tst_loader)*100))
    print('Adv Accuracy: \t {}'.format(adv_acc/len(tst_loader)*100))
    end = time.time()
    print('Test time: \t {}'.format(end-start))


if __name__ == "__main__":
    main()
