import torch
import torch.nn.functional as F
from torch import Tensor

# for type hint
from typing import Optional


def reduce_tensor(inputs: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return torch.mean(inputs)

    elif reduction == 'sum':
        return torch.sum(inputs)

    return inputs


def bha_coeff_log_prob(log_p: Tensor, log_q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of log(p) and log(q); the more similar the larger the coefficient
    :param log_p: (batch_size, num_classes) first log prob distribution
    :param log_q: (batch_size, num_classes) second log prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    # numerical unstable version
    # coefficient = torch.sum(torch.sqrt(p * q), dim=dim)
    # numerical stable version
    coefficient = torch.sum(torch.exp((log_p + log_q) / 2), dim=dim)

    return reduce_tensor(coefficient, reduction)


def bha_coeff(p: Tensor, q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param p: (batch_size, num_classes) first prob distribution
    :param q: (batch_size, num_classes) second prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_p = torch.log(p)
    log_q = torch.log(q)

    return bha_coeff_log_prob(log_p, log_q, dim=dim, reduction=reduction)


def bha_coeff_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param logits: (batch_size, num_classes) model predictions of the data
    :param targets: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_probs = F.log_softmax(logits, dim=dim)
    log_targets = torch.log(targets)

    # since BC(P,Q) is maximized when P and Q are the same, we minimize 1 - B(P,Q)
    return 1. - bha_coeff_log_prob(log_probs, log_targets, dim=dim, reduction=reduction)


def get_pair_indices(inputs: Tensor, ordered_pair: bool = False) -> Tensor:
    """
    Get pair indices between each element in input tensor
    Args:
        inputs: input tensor
        ordered_pair: if True, will return ordered pairs. (e.g. both inputs[i,j] and inputs[j,i] are included)
    Returns: a tensor of shape (K, 2) where K = choose(len(inputs),2) if ordered_pair is False.
        Else K = 2 * choose(len(inputs),2). Each row corresponds to two indices in inputs.
    """
    indices = torch.combinations(torch.tensor(range(len(inputs))), r=2)

    if ordered_pair:
        # make pairs ordered (e.g. both (0,1) and (1,0) are included)
        indices = torch.cat((indices, indices[:, [1, 0]]), dim=0)

    return indices


# https://github.com/zijian-hu/SimPLE/blob/5c9f69f57d42dea14a903adcd75c266c938d83de/loss/loss.py
def softmax_cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = 'mean') -> Tensor:
    """
    :param logits: (labeled_batch_size, num_classes) model output of the labeled data
    :param targets: (labeled_batch_size, num_classes) labels distribution for the data
    :param dim: the dimension or dimensions to reduce
    :param reduction: choose from 'mean', 'sum', and 'none'
    :return:
    """
    loss = -torch.sum(F.log_softmax(logits, dim=dim) * targets, dim=dim)

    return reduce_tensor(loss, reduction)


def mse_loss(prob: Tensor, targets: Tensor, reduction: str = 'mean', **kwargs) -> Tensor:
    return F.mse_loss(prob, targets, reduction=reduction)


class UnsupervisedLoss:
    def __init__(self,
                 reduction: str = "mean"):
        # if loss_type in ["entropy", "cross entropy"]:
        #     self.loss_use_prob = False
        #     self.loss_fn = softmax_cross_entropy_loss
        # else:
        self.loss_use_prob = True
        self.loss_fn = mse_loss

        # self.loss_thresholded = loss_thresholded
        # self.confidence_threshold = confidence_threshold
        self.reduction = reduction

    def __call__(self, probs: Tensor, targets: Tensor) -> Tensor:
        loss_input = probs
        loss = self.loss_fn(loss_input, targets, dim=1, reduction="none")

        # if self.loss_thresholded:
        #     targets_mask = (targets.max(dim=1).values > self.confidence_threshold)

        #     if len(loss.shape) > 1:
        #         # mse_loss returns a matrix, need to reshape mask
        #         targets_mask = targets_mask.view(-1, 1)

        #     loss *= targets_mask.float()

        return reduce_tensor(loss, reduction=self.reduction)





class PairLoss:
    def __init__(self,
                 reduction: str = "mean"):
        self.confidence_threshold = 0.95
        self.similarity_threshold = 0.9

        self.reduction = reduction

        self.similarity_metric = bha_coeff
        self.distance_loss_metric = bha_coeff_loss

    def __call__(self,
                 logits: Tensor,
                 probs: Tensor,
                 targets: Tensor,
                 *args,
                 indices: Optional[Tensor] = None,
                 **kwargs) -> Tensor:
        """
        Args:
            logits: (batch_size, num_classes) predictions of batch data
            probs: (batch_size, num_classes) softmax probs logits
            targets: (batch_size, num_classes) one-hot labels
        Returns: Pair loss value as a Tensor.
        """
        if indices is None:
            indices = get_pair_indices(targets, ordered_pair=True)
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]
        targets_max_prob = targets.max(dim=1).values

        return self.compute_loss(logits_j=logits[j_indices],
                                 probs_j=probs[j_indices],
                                 targets_i=targets[i_indices],
                                 targets_j=targets[j_indices],
                                 targets_i_max_prob=targets_max_prob[i_indices],
                                 total_size=total_size)

    def compute_loss(self,
                     logits_j: Tensor,
                     probs_j: Tensor,
                     targets_i: Tensor,
                     targets_j: Tensor,
                     targets_i_max_prob: Tensor,
                     total_size: int):
        # conf_mask should not track gradient
        conf_mask = (targets_i_max_prob > self.confidence_threshold).detach().float()

        similarities: Tensor = self.get_similarity(targets_i=targets_i,
                                                   targets_j=targets_j,
                                                   dim=1)
        # sim_mask should not track gradient
        sim_mask = F.threshold(similarities, self.similarity_threshold, 0).detach()

        distance = self.get_distance_loss(logits=logits_j,
                                          probs=probs_j,
                                          targets=targets_i,
                                          dim=1,
                                          reduction='none')

        loss = conf_mask * sim_mask * distance

        if self.reduction == "mean":
            loss = torch.sum(loss) / total_size
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

    def get_similarity(self,
                       targets_i: Tensor,
                       targets_j: Tensor,
                       *args,
                       **kwargs) -> Tensor:
        x, y = targets_i, targets_j

        return self.similarity_metric(x, y, *args, **kwargs)

    def get_distance_loss(self,
                          logits: Tensor,
                          probs: Tensor,
                          targets: Tensor,
                          *args,
                          **kwargs) -> Tensor:
        # if self.distance_loss_type == "prob":
        #     x, y = probs, targets
        # else:
        x, y = logits, targets

        return self.distance_loss_metric(x, y, *args, **kwargs)