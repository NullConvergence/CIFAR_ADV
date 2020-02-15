import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from cifar_data import get_datasets, mean, std
from custom_preproc.model_preproc import PreprocessingModel
from logger import Logger
import utils
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
                                         apply_transform=False)
    # initialization
    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model']).to(device)
    tmean = torch.tensor(mean).view(3, 1, 1).to(device)
    tstd = torch.tensor(std).view(3, 1, 1).to(device)
    preproc_model = PreprocessingModel(
        model, preproc={'mean': tmean, 'std': tstd})
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])

    scheduler = utils.get_scheduler(
        opt, cnfg['train'], cnfg['train']['epochs']*len(tr_loader))

    # train+test
    for epoch in range(cnfg['train']['epochs']):
        preproc_model.train()
        acc, tr_loss = 0, 0
        # train clean
        for _, (x, y_) in enumerate(tqdm(tr_loader)):
            x, y_ = x.to(device), y_.to(device)
            output = preproc_model(x)
            loss = criterion(output, y_)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc += (output.max(1)[1] == y_).sum().item() / len(y_)
            tr_loss += loss.item()
            scheduler.step()
        logger.log_train(epoch, tr_loss/len(tr_loader), acc/len(tr_loader)*100)
        # test
        if (epoch+1) % cnfg['test'] == 0 or epoch == 0:
            preproc_model.eval()
            tst_loss, tst_acc = 0, 0
            for _, (x, y_) in enumerate(tqdm(tst_loader)):
                x, y_ = x.to(device), y_.to(device)
                out = preproc_model(x)
                loss = criterion(out, y_)
                tst_acc += (out.max(1)[1] == y_).sum().item() / len(y_)
                tst_loss += loss.item()
            logger.log_test(epoch, tst_loss/len(tst_loader),
                            tst_acc/len(tst_loader)*100)

        # save
        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' \
                + cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)


if __name__ == "__main__":
    main()
