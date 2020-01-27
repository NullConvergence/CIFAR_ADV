import apex.amp as amp
import torch
from tqdm import tqdm
import utils


def train(epoch, model, criterion, opt, scheduler,
          tr_loader, device, logger, schdl_type='cyclic'):
    model.train()
    ep_loss = 0
    ep_acc = 0
    print('[INFO][TRAINING][clean_training] \t Epoch {} started.'.format(epoch))
    for batch_idx, (inpt, targets) in enumerate(tqdm(tr_loader)):
        inpt, targets = inpt.to(device), targets.to(device)
        output = model(inpt)
        loss = criterion(output, targets)
        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        ep_loss += loss.item()
        ep_acc += (output.max(1)[1] == targets).sum().item() / len(targets)
        if schdl_type == 'cyclic':
            utils.adjust_lr(opt, scheduler, logger, epoch*batch_idx)
    if schdl_type != 'cyclic':
        utils.adjust_lr(opt, scheduler, logger, epoch)
    logger.log_train(epoch, ep_loss/len(tr_loader),
                     (ep_acc/len(tr_loader))*100, "clean_training")


def test(epoch, model, tst_loader,  criterion, device, logger):
    tst_loss, tst_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for _, (inpt, targets) in enumerate(tst_loader):
            inpt, targets = inpt.to(device), targets.to(device)
            output = model(inpt)
            loss = criterion(output, targets)
            tst_loss += loss.item()
            tst_acc += (output.max(1)[1] ==
                        targets).sum().item() / len(targets)
    logger.log_test(epoch, tst_loss/len(tst_loader),
                    (tst_acc/len(tst_loader))*100, "clean_testing")
