import apex.amp as amp
import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils
from cifar_data import get_classes

text_classes = get_classes()


def train(epoch, model, word2vec, opt, scheduler,
          tr_loader, device, logger):
    model.train()
    ep_loss = 0
    ep_acc = 0
    print('[INFO][TRAINING][clean_training] \t Epoch {} started.'.format(epoch))
    for batch_idx, (inpt, targets) in enumerate(tqdm(tr_loader)):
        inpt, targets = inpt.to(device), targets.to(device)
        vec_targets = torch.tensor(get_proto_targets(
            targets, word2vec, text_classes)).to(device)
        output = model(inpt)
        loss = cosine_loss(output, vec_targets)
        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        ep_loss += loss.item()
        classes = torch.tensor(get_normal_targets(
            output, word2vec, text_classes, device)).to(device)
        ep_acc += torch.sum(classes == targets).item() / len(targets)
        utils.adjust_lr(opt, scheduler, logger, epoch*batch_idx, do_log=False)

    utils.log_lr(logger, opt, epoch)
    logger.log_train(epoch, ep_loss/len(tr_loader),
                     (ep_acc/len(tr_loader))*100, "clean_training")


def test(epoch, model, word2vec, tst_loader, device, logger):
    tst_loss, tst_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for _, (inpt, targets) in enumerate(tst_loader):
            inpt, targets = inpt.to(device), targets.to(device)
            vec_targets = torch.tensor(get_proto_targets(
                targets, word2vec, text_classes)).to(device)
            output = model(inpt)
            loss = cosine_loss(output, vec_targets)
            tst_loss += loss.item()
            classes = torch.tensor(get_normal_targets(
                output, word2vec, text_classes, device)).to(device)
            tst_acc += torch.sum(classes == targets).item() / len(targets)
    logger.log_test(epoch, tst_loss/len(tst_loader),
                    (tst_acc/len(tst_loader))*100, "clean_testing")


def cosine_loss(output, target, dim=1, eps=1e-8, reduction="mean"):
    sim = 1 - F.cosine_similarity(output, target, dim, eps)
    if reduction == "mean":
        return sim.mean()
    else:
        return sim


def get_proto_targets(targets, word2vec, text_classes):
    text_target = [text_classes[i] for i in targets]
    vec_target = [word2vec.wv[t] for t in text_target]
    return vec_target


def get_normal_targets(batch, word2vec, text_classes, dev):
    classes = []
    for _, output_vector in enumerate(batch):
        class_vec = torch.tensor([word2vec.wv[t]
                                  for t in text_classes]).to(dev)
        best = 10  # TODO: remmember why is this 10? :-)
        index = 0
        for j, vec in enumerate(class_vec):
            sim = 1 - F.cosine_similarity(output_vector, vec, 0)
            if sim < best:
                best = sim
                index = j
        classes.append(index)
    return classes
