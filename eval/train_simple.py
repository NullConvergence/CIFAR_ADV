import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from cifar_data import get_datasets
from logger import Logger
import utils
import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eval/cnfg.yml', type=str)
    return parser.parse_args()


def pgd_attack(model, images, labels, device, eps=8/255, alpha=2/255, iters=40):
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = F.cross_entropy(outputs, labels).cuda()
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
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model'])
    model = nn.DataParallel(model)
    model.cuda()

    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])
    scheduler = utils.get_scheduler(
        opt, cnfg['train'], cnfg['train']['epochs']*len(tr_loader))

    def test():
        acc, adv_acc = 0, 0
        for batch_idx, (x, y_) in enumerate(tqdm(tst_loader)):
            x, y_ = x.cuda(), y_.cuda()
            adv = pgd_attack(model, x, y_, device, eps=cnfg['pgd']['epsilon']/255,
                             alpha=cnfg['pgd']['alpha']/255,
                             iters=cnfg['pgd']['iter'])
            with torch.no_grad():
                out = model(x)
                adv_out = model(adv)
                acc += (out.max(1)[1] == y_).sum().item() / len(y_)
                adv_acc += (adv_out.max(1)[1] == y_).sum().item() / len(y_)

        return acc/len(tst_loader), adv_acc / len(tst_loader)

    def train():
        for epoch in range(cnfg['train']['epochs']):
            print('Epoch: ', epoch)
            model.train()
            adv_acc = 0
            for batch_idx, (x, y_) in enumerate(tqdm(tr_loader)):
                x, y_ = x.cuda(), y_.cuda()
                if (epoch+1) <= cnfg['mixed']['adv_epochs']:
                    adv = pgd_attack(model, x, y_, device, eps=cnfg['pgd']['epsilon']/255,
                                     alpha=cnfg['pgd']['alpha']/255,
                                     iters=cnfg['pgd']['iter'])

                    adv_out = model(adv)
                    loss = F.cross_entropy(adv_out, y_)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    adv_acc += (adv_out.max(1)[1] == y_).sum().item() / len(y_)
                else:
                    out = model(x)
                    loss = F.cross_entropy(out, y_)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    adv_acc += (out.max(1)[1] == y_).sum().item() / len(y_)
                scheduler.step()
            logger.log_train(epoch, 0, adv_acc / len(tr_loader), 'train')

            if (epoch+1) % cnfg['test'] == 0:
                t_acc, t_adv = test()
                logger.log_test(epoch, 0, t_acc, 't_acc')
                logger.log_test_adversarial(epoch, 0, t_adv, 't_adv')
            if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
                pth = 'models/' + cnfg['logger']['project'] + '_' \
                    + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
                utils.save_model(model, cnfg, epoch, pth)
                logger.log_model(pth)

    train()


if __name__ == "__main__":
    main()
