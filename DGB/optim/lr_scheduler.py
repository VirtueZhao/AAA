"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch

AVAILABLE_LR_SCHEDULERS = ["single_step", "cosine"]


def build_lr_scheduler(optimizer, optim_cfg):
    """A Function Wrapper for Building a Learning Rate Scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler_name = optim_cfg.LR_SCHEDULER
    step_size = optim_cfg.STEP_SIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler_name not in AVAILABLE_LR_SCHEDULERS:
        raise ValueError("Unsupported LR Scheduler: {}. Current Support LR Schedulers: {}.".format(lr_scheduler_name, AVAILABLE_LR_SCHEDULERS))

    if lr_scheduler_name == "single_step":
        if step_size <= 0:
            step_size = max_epoch
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_name == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)

    return lr_scheduler

