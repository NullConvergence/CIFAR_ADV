import apex.amp as amp
import argparse
import torch
import torch.nn as nn
import torchvision
from cifar_data import get_datasets
import pgd.pgd_trainer as pgd
import clean.trainer as clean
from logger import Logger
import utils
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
import numpy as np
from cifar_data import mean, std

mnn = np.asarray(mean).reshape((3, 1, 1))
stdd = np.asarray(std).reshape((3, 1, 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eval/cnfg.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data
    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'])

    # initialization
    utils.set_seed(cnfg['seed'])
    device = device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    # logger = Logger(cnfg)
    model = utils.get_model(cnfg['model'])
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])

    checkpoint = torch.load(cnfg['resume']['path'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # art
    classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=criterion,
                                   optimizer=opt, input_shape=[128, 3, 32, 32],
                                   nb_classes=10)
    attack = FastGradientMethod(classifier=classifier, eps=8./255)

    acc = 0
    for batch_idx, (x, y_) in enumerate(tst_loader):
        x_np, y_np_ = x.numpy(), y_.numpy()
        x_adv = attack.generate(x=x_np)
        preds = np.argmax(classifier.predict(x_adv), axis=1)
        acc_batch = np.sum(preds == y_np_) / y_np_.shape[0]
        acc += acc_batch
        print('Batch {} accuraccy: \t {}'.format(batch_idx, acc_batch))
    print('Adversarial accuracy: \t {}'.format(acc / len(tst_loader)))


if __name__ == '__main__':
    main()
