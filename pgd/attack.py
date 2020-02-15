import apex.amp as amp
import torch
import torch.nn.functional as F
from cifar_data import mean, std


def train_pgd(model, device, criterion, inpt, target, epsilon, alpha,
              iter, restart, opt, d_init='',
              l_limit=0, u_limit=1, doamp=True):
    epsilon, alpha = get_eps_alph(epsilon, alpha, device)
    # init delta
    delta = torch.zeros_like(inpt).to(device)
    if d_init == 'random':
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(),
                                       epsilon[0][0][0].item())
        delta.data = clamp(delta, l_limit, u_limit)
    delta.requires_grad = True
    # perform internal max loop
    for iter in range(iter):
        output = model(inpt+delta)
        loss = criterion(output, target)
        if doamp is True:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(
            delta+alpha*torch.sign(grad), -epsilon, epsilon)
        delta.grad.zero_()
    delta = delta.detach()
    return delta


def eval_pgd(model, device, criterion, inpt, target, epsilon, alpha, iter,
             restarts, l_limit, u_limit, opt, doamp=True):
    max_loss = torch.zeros(target.shape[0]).to(device)
    max_delta = torch.zeros_like(inpt).to(device)
    epsilon, alpha = get_eps_alph(epsilon, alpha, device)

    for i in range(restarts):
        delta = torch.zeros_like(inpt).to(device)
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-epsilon[j][0]
                                       [0].item(), epsilon[0][0][0].item())
        delta.data = clamp(delta, l_limit - inpt, u_limit - inpt)
        delta.requires_grad = True

        for _ in range(iter):
            output = model(inpt+delta)
            # perturb only inputs that get correctly
            # classified
            index = torch.where(output.max(1)[1] == target)
            loss = F.cross_entropy(output, target)
            # loss = criterion(output, target)
            if doamp is True:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, l_limit -
                      inpt[index[0], :, :, :], u_limit - inpt[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(
            model(inpt+delta), target, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# TODO: the following methods are not efficiently called
# (premature optimization is currently the root of all evil)


def get_limits(dev):
    mu = torch.tensor(mean).view(3, 1, 1).to(dev)
    stdd = torch.tensor(std).view(3, 1, 1).to(dev)
    upper_limit = ((1. - mu) / stdd)
    lower_limit = ((0. - mu) / stdd)
    return lower_limit, upper_limit


def get_eps_alph(epsilon, alpha, dev):
    stdd = torch.tensor(std).view(3, 1, 1).to(dev)
    eps = (epsilon / 255.) / stdd
    alph = (alpha / 255.) / stdd
    return eps, alph
