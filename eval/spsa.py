"""
" Modified with love from: https://github.com/tensorflow/cleverhans/pull/1086
"""
import numpy as np
import torch
from torch import optim


def spsa(model_fn, x, eps, nb_iter, clip_min=-np.inf, clip_max=np.inf, y=None,
         targeted=False, early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01,
         spsa_samples=128, spsa_iters=1, is_debug=True, sanity_checks=True):

    # If a data range was specified, check that the input was in that range
    print(clip_min, x)
    assert torch.all(x >= clip_min)
    assert torch.all(x <= clip_max)

    if is_debug:
        print("Starting SPSA attack with eps = {}".format(eps))

    perturbation = (torch.rand_like(x) * 2 - 1) * eps
    _project_perturbation(perturbation, eps, x, clip_min, clip_max)
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for i in range(nb_iter):
        def loss_fn(pert):
            """
            Margin logit loss, with correct sign for targeted vs untargeted loss.
            """
            logits = model_fn(x + pert)
            loss_multiplier = 1 if targeted else -1
            return loss_multiplier * _margin_logit_loss(logits, y.expand(len(pert)))

        spsa_grad = _compute_spsa_gradient(loss_fn, x, delta=delta,
                                           samples=spsa_samples, iters=spsa_iters)
        perturbation.grad = spsa_grad
        optimizer.step()

        _project_perturbation(perturbation, eps, x, clip_min, clip_max)

        loss = loss_fn(perturbation).item()
        if is_debug:
            print('Iteration {}: loss = {}'.format(i, loss))
        if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:
            break

    adv_x = (x + perturbation).detach()

    asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
    asserts.append(torch.all(adv_x >= clip_min))
    asserts.append(torch.all(adv_x <= clip_max))

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def _project_perturbation(perturbation, epsilon, input_image, clip_min=-np.inf,
                          clip_max=np.inf):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation,
                            clip_min, clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())

    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = torch.mean(
            loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta
        grad_list.append(avg_grad)

    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)


def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.

    The loss is high when `label` is unlikely (targeted by default).
    """

    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float('-inf')),
        logits,
    )
    max_incorrect_logits, _ = torch.max(
        incorrect_logits, 1)

    return max_incorrect_logits - correct_logits
