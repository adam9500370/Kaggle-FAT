import torch
import torch.nn.functional as F


def sigmoid_focal_loss_with_logits(input, target, reduction='mean', alpha=1.0, gamma=2.0):
    """
    input: (N, *)
    target: (N, *)
    """
    ## Ref: https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
    # A numerically stable implementation of modulator.
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    modulator = 1.0 if gamma == 0.0 else torch.exp(-gamma * target * input - gamma * torch.log1p(torch.exp(-1.0 * input)))
    loss = modulator * bce_loss

    loss = (alpha * target).sum(dim=1, keepdim=True) * loss

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        # Normalize by the total number of positive samples.
        loss = loss.sum() / target.sum()
    return loss
