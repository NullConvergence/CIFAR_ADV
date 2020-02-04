import apex.amp as amp
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pgd.attack import clamp, get_limits
import utils


def train(epoch, delta, model, criterion, opt, scheduler, cnfg,
          tr_loader, device, logger, epsilon=8/255, alpha=2/255, schdl_type='cyclic'):
    model.train()
    ep_loss, ep_acc = 0, 0
    l_limit, u_limit = get_limits(device)
    print('[INFO][TRAINING][fast_adv_training] \t Epoch {} started.'.format(epoch))
    for batch_idx, (inpt, targets) in enumerate(tqdm(tr_loader)):
        inpt, targets = inpt.to(device), targets.to(device)
        if cnfg['pgd']['delta_init'] != 'previous':
            delta = torch.zeros_like(inpt).cuda
        if cnfg['pgd']['delta_init'] == 'random':
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i]
                                           [0][0].item(), epsilon[0][0][0].item())
            delta.data = clamp(delta, l_limit-inpt, u_limit-inpt)
        delta.requres_gard = True
        output = model(inpt+delta[:inpt.size(0)])
        loss = F.cross_entropy(output, targets)
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(
            delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:inpt.size(0)] = clamp(
            delta[inpt.size(0)], l_limit-inpt, u_limit-inpt)
        delta = delta.detach()
        output = model(inpt+delta[:inpt.size(0)])
        loss = criterion(output, targets)
        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        utils.adjust_lr(opt, scheduler, logger,
                        epoch*batch_idx, do_log=False)

        ep_loss += loss.item()  # * targets.size(0)
        ep_acc += (output.max(1)[1] == targets).sum().item() / len(targets)

    logger.log_train(epoch, ep_loss/len(tr_loader),
                     (ep_acc/len(tr_loader))*100, "fast_adv_training")
    return delta
