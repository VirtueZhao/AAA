"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import errno
import torch
import random
import warnings
import numpy as np
import os.path as osp
import torch.nn as nn
from difflib import SequenceMatcher
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def list_non_hidden_directory(path, sort=False):
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def check_is_file(file_path):
    is_file = osp.isfile(file_path)
    if not is_file:
        warnings.warn("No File Found at {}".format(file_path))
    return is_file


def init_network_weights(model, init_type="normal", gain=0.02):
    def _init_func(m):
        class_name = m.__class__.__name__

        if hasattr(m, "weight") and (
            class_name.find("Conv") != -1 or class_name.find("Linear") != -1
        ):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif class_name.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif class_name.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def count_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_top_k_accuracy(output, class_label, top_k=1):
    batch_size = class_label.size(0)
    top_k_predictions = output.topk(top_k, 1, True, True)[1].t()
    class_label = class_label.view(1, -1).expand_as(top_k_predictions)

    correct = top_k_predictions.eq(class_label).float().sum(0, keepdim=True)
    results = []
    for k in range(top_k):
        correct_k = correct[k].float().sum(0, keepdim=True)
        accuracy = 100.0 * correct_k / batch_size
        results.append(accuracy)

    return results


def compute_gradients_length(gradients, channel=False):
    gradients_length = []
    for current_gradient in gradients:
        current_gradient = current_gradient.cpu().numpy()
        if channel:
            length = 0
            for g in current_gradient:
                g = g.flatten()
                length += np.linalg.norm(g)
            length = length / len(current_gradient)
        else:
            g = current_gradient.flatten()
            length = np.linalg.norm(g)
        gradients_length.append(length)

    return torch.tensor(gradients_length).to(gradients.device)


def gini_coefficient(embedding):
    # embedding = embedding.cpu()
    embedding = torch.sort(embedding)[0]
    n = embedding.shape[0]

    # # Calculate the Gini coefficient
    index = torch.arange(1, n + 1, dtype=torch.float32)
    gini = (torch.sum((2 * index - n - 1) * embedding)) / (n * torch.sum(embedding))

    return gini


def measure_diversity(embeddings, diversity_type):
    # print("Measure Diversity: {}".format(diversity_type))
    if diversity_type == "gini":
        gini_values = torch.zeros(embeddings.shape[0])
        for i in range(embeddings.shape[0]):
            gini_values[i] = gini_coefficient(embeddings[i])

        return gini_values
    elif diversity_type == "euclidean":
        embeddings = embeddings.detach().numpy()
        kmeans = KMeans(n_clusters=1, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        centroid = kmeans.cluster_centers_[0]
        euclidean_distances = torch.Tensor(
            pairwise_distances(embeddings, [centroid])
        ).reshape(-1)

        distances_min = torch.min(euclidean_distances)
        distances_max = torch.max(euclidean_distances)
        normalized_distances = (euclidean_distances - distances_min) / (
            distances_max - distances_min
        )

        return normalized_distances
    elif diversity_type == "cosine":
        embeddings = embeddings.detach().numpy()
        kmeans = KMeans(n_clusters=1, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        centroid = kmeans.cluster_centers_[0]
        embeddings = torch.Tensor(embeddings)
        centroid = torch.Tensor(centroid)
        cosine_similarity = torch.nn.functional.cosine_similarity(embeddings, centroid)

        return 1 - cosine_similarity
    else:
        raise NotImplementedError


def compute_impact_factor(diversity, lower_bound, upper_bound, individual_factor=False):
    lmda = 1 - diversity
    # lmda = diversity
    lmda_min = torch.min(lmda)
    lmda_max = torch.max(lmda)
    normalized_lmda = lower_bound + (
        (lmda - lmda_min) * (upper_bound - lower_bound)
    ) / (lmda_max - lmda_min)

    if individual_factor:
        return normalized_lmda
    else:
        return normalized_lmda.mean().item()
