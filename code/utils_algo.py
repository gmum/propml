import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def compute_pml_loss(output, target, noise_rate):
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # first part - all labels
    loss_mat_pos = criterion(output, target)
    nr_mat_neg = torch.FloatTensor(target.size()).cuda()
    nr_mat_neg[target == 0] = 1
    nr_mat_neg[target == 1] = 1 - noise_rate
    first_part = nr_mat_neg.mul(loss_mat_pos)

    # second part - only negative labels
    loss_mat_neg = criterion(output, 1 - target)
    nr_mat_pos = torch.FloatTensor(target.size()).cuda()
    nr_mat_pos[target == 0] = noise_rate
    nr_mat_pos[target == 1] = 0
    second_part = nr_mat_pos.mul(loss_mat_neg)

    loss = first_part - second_part

    C = 1 / (1 - noise_rate)
    loss = loss.mul(C).mean()
    loss = torch.clamp(loss, min=0.0)

    return loss


def compute_ub_loss(output, target, noise_rate, relu):
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_mat_pos = criterion(output, target)
    loss_mat_neg = criterion(output, 1 - target)

    if isinstance(noise_rate, list):
        neg_nr, pos_nr = noise_rate

        nr_mat_pos = torch.FloatTensor(target.size()).cuda()
        nr_mat_pos[target != 1] = neg_nr
        nr_mat_pos[target == 1] = pos_nr
        nr_mat_neg = torch.FloatTensor(target.size()).cuda()
        nr_mat_neg[1 - target != 1] = neg_nr
        nr_mat_neg[1 - target == 1] = pos_nr
        C = 1 / (1 - neg_nr - pos_nr)

    if relu:
        loss = torch.clamp(((1 - nr_mat_neg).mul(loss_mat_pos) - nr_mat_pos.mul(loss_mat_neg)).mul(C).mean(), min=0.0)
    else:
        loss = ((1 - nr_mat_neg).mul(loss_mat_pos) - nr_mat_pos.mul(loss_mat_neg)).mul(C).mean()

    return loss


class ProPaLL(nn.Module):
    def __init__(self, reduce="sum"):
        super().__init__()
        self.reduce = reduce

    def forward(self, inputs, target):
        tmp = inputs.unsqueeze(1)
        tmp = torch.cat((torch.zeros_like(tmp), tmp), dim=1)
        tmp = torch.logsumexp(tmp, dim=1, keepdim=False)

        second_part = ((1 - target) * tmp).sum(1)
        # target_bool = target.bool()
        # second_part = torch.where(target_bool, 0, tmp).sum(1)
        assert torch.isfinite(second_part).all(), f"second_part: [{second_part}]"

        temp = torch.where(
            target.bool(),
            inputs,
            torch.as_tensor(torch.finfo(inputs.dtype).min, dtype=inputs.dtype, device=inputs.device),
        )
        max_temp = torch.max(temp, dim=1, keepdim=False).values

        first_part = torch.where(
            torch.gt(max_temp, -10),
            torch.log(
                1
                - torch.sum(target * tmp, dim=1).neg().exp()
                + torch.as_tensor(torch.finfo(inputs.dtype).eps, dtype=inputs.dtype, device=inputs.device)
            ),
            # torch.log(1 - torch.sum(target * tmp, dim=1).neg().exp()),
            torch.logsumexp(temp, dim=1, keepdim=False),
        )
        # first_part = torch.log(1 - torch.sum(target * tmp, dim=1).neg().exp())
        assert torch.isfinite(first_part).all(), f"first_part: [{first_part}]"

        results = torch.add(second_part, first_part, alpha=-1)
        if self.reduce == "mean":
            results = results.mean()
        return results


class PMLL(nn.Module):
    def __init__(self, reduce="sum", alpha=1.):
        super().__init__()
        self.reduce = reduce
        self.alpha = alpha

    def forward(self, inputs, target):
        tmp = inputs.unsqueeze(1)
        tmp = torch.cat((torch.zeros_like(tmp), tmp), dim=1)
        tmp = torch.logsumexp(tmp, dim=1, keepdim=False)

        second_part = ((1 - target) * tmp).sum(1)
        assert torch.isfinite(second_part).all(), f"second_part: [{second_part}]"

        temp = torch.where(
            target.bool(),
            inputs,
            torch.as_tensor(torch.finfo(inputs.dtype).min, dtype=inputs.dtype, device=inputs.device),
        )
        max_temp = torch.max(temp, dim=1, keepdim=True).values
        tmp = torch.sub(max_temp, target * inputs, alpha=1).unsqueeze(1)
        tmp = torch.cat((max_temp.tile(1, tmp.shape[-1]).unsqueeze(1), tmp), dim=1)
        max_temp = max_temp.squeeze()
        tmp = torch.logsumexp(tmp, dim=1, keepdim=False).neg().exp()
        first_part_II = torch.sum(target * tmp, dim=1).log().add(max_temp)
        first_part_I = torch.sum(target * torch.sigmoid(inputs), dim=1).log()
        first_part = torch.where(torch.gt(max_temp, -10), first_part_I, first_part_II)
        assert torch.isfinite(first_part).all(), f"first_part: [{first_part} "

        results = torch.add(second_part * self.alpha, first_part, alpha=-1)
        if self.reduce == "mean":
            results = results.mean()
        return results, first_part.mean(), second_part.mean()


def precision_recall_f1(test_target, scores):
    '''
    Evaluate the average per-class precision (CP), recall (CR), F1 (CF1)
    and the average overall precision (OP), recall (OR), F1 (OR1)
    '''

    assert scores.shape == test_target.shape

    threshold_level = 2

    N, C = scores.shape

    scores_sorted = -np.sort(-scores, axis=1)
    threshold = scores_sorted[:, threshold_level].reshape(N, -1)

    # threshold = 0.6
    pred_target = (scores >= threshold).astype(np.float64)

    N_g = np.sum(test_target, axis=0).astype(np.float64)
    N_p = np.sum(pred_target, axis=0).astype(np.float64)
    N_c = np.sum(test_target * pred_target, axis=0).astype(np.float64)

    OP = np.sum(N_c) / np.sum(N_p)
    OR = np.sum(N_c) / np.sum(N_g)
    OF1 = (2 * OP * OR) / (OP + OR)

    N_p[N_p == 0] = 1

    CP = np.sum(N_c / N_p) / C
    CR = np.sum(N_c / N_g) / C
    CF1 = (2 * CP * CR) / (CP + CR)

    return OF1 * 100, CF1 * 100


def precision_recall_f1_scikit(test_target, scores):
    '''
    Evaluate the average per-class precision (CP), recall (CR), F1 (CF1)
    and the average overall precision (OP), recall (OR), F1 (OR1)
    '''

    assert scores.shape == test_target.shape

    threshold = 0.6

    OF1 = f1_score(test_target, scores > threshold, average='micro')
    CF1 = f1_score(test_target, scores > threshold, average='macro')

    return OF1 * 100, CF1 * 100
