
import torch
import torch.nn as nn
import torch.nn.functional as F


class FAST(nn.Module):
    def __init__(self, model, cnfg, normalization=255, clip_min=0, clip_max=1):
        super(FAST, self).__init__()
        self.model = model
        self.rand_init = cnfg['delta-init']
        self.eps = cnfg['epsilon'] / normalization
        self.iter = cnfg['iter']
        self.alpha = cnfg['alpha'] / normalization
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x, y_):
        x_ = x.detach()
        if self.rand_init == 'random':
            x_ = x_ + torch.zeros_like(x_).uniform_(-self.eps, self.eps)

        with torch.enable_grad():
            logits = self.model(x_)
            loss = F.cross_entropy(logits, y_)
        grad = torch.autograd.grad(loss, [x_])[0]
        x_ = x_.detach() + self.alpha*torch.sign(grad.detach())
        x_ = torch.min(torch.max(x_, x-self.eps), x+self.eps)
        x_ = torch.clamp(x_, self.clip_min, self.clip_max)

        return self.model(x_), x_
