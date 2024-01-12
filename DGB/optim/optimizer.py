"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
import torch.nn as nn

AVAILABLE_OPTIMIZERS = ["SGD", "ADAM"]


def build_optimizer(model, optim_cfg):
    """A Function Wrapper for Building an Optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """
    optimizer_name = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPENING
    sgd_nesterov = optim_cfg.SGD_NESTEROV

    if optimizer_name not in AVAILABLE_OPTIMIZERS:
        raise ValueError("Unsupported Optimizer: {}. Current Support Optimizers: {}.".format(optimizer_name, AVAILABLE_OPTIMIZERS))

    if isinstance(model, nn.Module):
        params = model.parameters()
    else:
        params = model

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            params=params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov
        )
    # elif optimizer_name == "ADAM":
    #     optimizer = torch.optim.Adam(
    #         params=params,
    #         lr=lr,
    #         weight_decay=weight_decay,
    #         betas=(0.9, 0.999)
    #     )

    return optimizer
