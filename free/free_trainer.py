import apex.amp as amp
import torch
from tqdm import tqdm
from pgd.attack import clamp, get_limits
import utils


def train(epoch, delta, batch_runs, epsilon, model, criterion, opt, scheduler,
          tr_loader, device, logger):
    model.train()
    ep_loss, ep_acc = 0, 0
    l_limit, u_limit = get_limits(device)
    print('[INFO][TRAINING][free_adv_training] \t Epoch {} started.'.format(epoch))
    for batch_idx, (inpt, targets) in enumerate(tqdm(tr_loader)):
        inpt, targets = inpt.to(device), targets.to(device)
        for mini_batchidx in range(batch_runs):
            output = model(inpt + delta[:inpt.size(0)])
            loss = criterion(output, targets)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(
                delta + epsilon*torch.sign(grad), -epsilon, epsilon)
            delta.data[:inpt.size(0)] = clamp(delta[:inpt.size(0)],
                                              l_limit - inpt, u_limit - inpt)
            opt.step()
            delta.grad.zero_()
            utils.adjust_lr(opt, scheduler, logger,
                            epoch*batch_idx*mini_batchidx, do_log=False)
        ep_loss += loss.item()  # * targets.size(0)
        ep_acc += (output.max(1)[1] == targets).sum().item() / len(targets)

    utils.log_lr(logger, opt, epoch)
    logger.log_train(epoch, ep_loss/len(tr_loader),
                     (ep_acc/len(tr_loader))*100, "free_adv_training")
    return delta
