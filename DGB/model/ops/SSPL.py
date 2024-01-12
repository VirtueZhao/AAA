import torch
import torch.nn as nn
from DGB.utils.tools import compute_gradients_length


class SymmetricSelfPacedLearning(nn.Module):

    def __init__(self, cfg, current_epoch):
        super().__init__()
        self.cfg = cfg
        self.eta = 1
        self.current_epoch = current_epoch
        self.epoch_step_size = 2 / (self.cfg.OPTIM.MAX_EPOCH - 1)
        self.weight_first = 2 - self.current_epoch * self.epoch_step_size
        self.weight_last = 2 - self.weight_first

    def forward(self, loss, gradients, difficulty_type):
        if difficulty_type == "loss":
            example_difficulty = loss
        elif difficulty_type == "gradients":
            example_difficulty = compute_gradients_length(gradients)
        elif difficulty_type == "LGDM":
            gradients_length = compute_gradients_length(gradients)
            example_difficulty = 0.5 * loss + 0.5 * gradients_length
        else:
            raise NotImplementedError

        weight_matrix = self.compute_weight_matrix(example_difficulty)
        loss = loss * weight_matrix
        loss = loss.mean()

        return loss, example_difficulty

    def compute_weight_matrix(self, example_difficulty):
        batch_step_size = (self.weight_first - self.weight_last) / (len(example_difficulty) - 1)
        weight_matrix = torch.zeros_like(example_difficulty)
        indices = torch.argsort(example_difficulty)

        for i, index in enumerate(indices):
            weight_matrix[index] = self.weight_first - batch_step_size * i

        return weight_matrix


