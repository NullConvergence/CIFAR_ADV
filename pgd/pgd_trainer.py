import apex.amp as amp
import torch
from tqdm import tqdm
import pgd.attack as pgd
import utils


def train(epoch, model, criterion, opt, scheduler, cnfg,
          tr_loader, device, logger, doamp=True):
    model.train()
    ep_loss = 0
    ep_acc = 0
    print('[INFO][TRAINING][clean_training] \t Epoch {} started.'.format(epoch))
    l_limit, u_limit = pgd.get_limits(device)
    for batch_idx, (inpt, targets) in enumerate(tqdm(tr_loader)):
        inpt, targets = inpt.to(device), targets.to(device)
        delta = pgd.train_pgd(model, device, criterion, inpt, targets,
                              epsilon=cnfg['pgd']['epsilon'],
                              alpha=cnfg['pgd']['alpha'],
                              iter=cnfg['pgd']['iter'],
                              opt=opt,
                              restart=cnfg['pgd']['restarts'],
                              d_init=cnfg['pgd']['delta-init'],
                              l_limit=l_limit, u_limit=u_limit, doamp=doamp)
        output = model(inpt+delta)
        loss = criterion(output, targets)
        opt.zero_grad()
        if doamp == True:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()
        ep_loss += loss.item()
        ep_acc += (output.max(1)[1] == targets).sum().item() / len(targets)
        utils.adjust_lr(opt, scheduler, logger, epoch*batch_idx, do_log=False)

    utils.log_lr(logger, opt, epoch)
    logger.log_train(epoch, ep_loss/len(tr_loader),
                     (ep_acc/len(tr_loader))*100, "pgd_training")


def test(epoch, model, tst_loader,  criterion, device, logger, cnfg, opt, doamp=True):
    tst_loss, adv_loss, tst_acc, adv_acc = 0, 0, 0, 0
    model.eval()
    l_limit, u_limit = pgd.get_limits(device)
    for _, (inpt, targets) in enumerate(tst_loader):
        inpt, targets = inpt.to(device), targets.to(device)
        pgd_delta = pgd.eval_pgd(model, device, criterion, inpt, targets,
                                 cnfg['pgd']['epsilon'],
                                 cnfg['pgd']['alpha'],
                                 cnfg['pgd']['iter'],
                                 cnfg['pgd']['restarts'],
                                 l_limit, u_limit, opt, doamp=doamp)
        with torch.no_grad():
            # normal measurements
            output = model(inpt)
            loss = criterion(output, targets)
            tst_loss += loss.item()
            tst_acc += (output.max(1)[1] ==
                        targets).sum().item() / len(targets)
            # adversarial
            adv_output = model(inpt+pgd_delta)
            adv_loss = criterion(output, targets)
            adv_loss += adv_loss.item()
            adv_acc += (adv_output.max(1)[1] ==
                        targets).sum().item() / len(targets)

    logger.log_test(epoch, tst_loss/len(tst_loader),
                    (tst_acc/len(tst_loader))*100, "clean_testing")
    logger.log_test_adversarial(epoch, adv_loss/len(tst_loader),
                                (adv_acc/len(tst_loader))*100, "pgd_testing")
