import torch
from torch import nn


def compute_dist_mat(x, y):
    m, n = x.size(0), y.size(0)
    dist_mat = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)

    return dist_mat


def hard_example_mining(dist_mat, labels):
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLoss(nn.Module):

    def __init__(self, margin=None):
        super().__init__()
        self._margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    @property
    def margin(self):
        return self._margin

    def forward(self, representations, labels):
        # print("Computing Triplet Loss")
        dist_mat = compute_dist_mat(representations, representations)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss
