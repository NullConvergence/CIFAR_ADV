
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PGD(nn.Module):
    def __init__(self, model, cnfg, normalization=255, clip_min=0,
                 clip_max=1):
        super(PGD, self).__init__()
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
            x = torch.clamp(x, self.clip_min, self.clip_max)

        for _ in range(self.iter):
            x_.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x_)
                loss = F.cross_entropy(logits, y_)
            grad = torch.autograd.grad(loss, [x_])[0]
            x_ = x_.detach() + self.alpha*torch.sign(grad.detach())
            x_ = torch.min(torch.max(x_, x-self.eps), x+self.eps)
            x_ = torch.clamp(x_, self.clip_min, self.clip_max)
        return self.model(x_), x_


class PGDTest(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, epoch, logger, model, pgd, tst_loader, device):
        acc, loss, adv_acc, adv_loss = 0, 0, 0, 0
        model.eval()
        for batch_idx, (x, y_) in enumerate(tqdm(tst_loader)):
            x, y_ = x.to(device), y_.to(device)
            with torch.no_grad():
                out = model(x)
                loss += F.cross_entropy(out, y_)
                acc += (out.max(1)[1] == y_).sum().item() / len(y_)

                adv_output, x_ = pgd(x, y_)
                adv_loss += F.cross_entropy(adv_output, y_)
                adv_acc += (adv_output.max(1)[1] == y_).sum().item() / len(y_)

        logger.log_test(epoch, loss/len(tst_loader), acc/len(tst_loader)*100)
        logger.log_test_adversarial(
            epoch, adv_loss/len(tst_loader), adv_acc/len(tst_loader)*100)
