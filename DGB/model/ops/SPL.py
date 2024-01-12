import torch
import torch.nn as nn
from DGB.utils.tools import compute_gradients_length


class SelfPacedLearning(nn.Module):

    def __init__(self, lmda=0.1):
        super().__init__()
        self.lmda = lmda

    def forward(self, loss, gradients, difficulty_type):
        # print("Difficulty Type: {}".format(difficulty_type))
        if difficulty_type == "loss":
            example_difficulty = loss
        elif difficulty_type == "gradients":
            example_difficulty = compute_gradients_length(gradients)
        else:
            raise NotImplementedError

        weight_matrix = self.compute_weight_matrix(example_difficulty)
        loss = loss * weight_matrix
        loss = loss.sum() / torch.count_nonzero(loss)
        return loss

    def compute_weight_matrix(self, loss):
        weight_matrix = torch.zeros_like(loss)
        values, indices = torch.topk(loss, int(loss.numel() * self.lmda), largest=False)
        weight_matrix[indices] = 1

        return weight_matrix
