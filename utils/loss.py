import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# class DiversityLoss(nn.Module):
#     def __init__(self):
#         super(DiversityLoss, self).__init__()
#         self.criterion = nn.L1Loss()
#
#     def forward(self, logits):
#         logits = logits.var()
#         return self.criterion(logits, torch.zeros_like(logits))


# class UniformSkipLoss(nn.Module):
#     def __init__(self, target, blocks):
#         super(UniformSkipLoss, self).__init__()
#         self.criterion = nn.L1Loss()
#         self.target = torch.ones([blocks * 3], dtype=torch.float).cuda() * target / blocks
#
#     def forward(self, logits):
#         return self.criterion(logits, self.target)


class FLOPSL1Loss(nn.Module):
    def __init__(self, target=0.0):
        super(FLOPSL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.target_num = target

    def forward(self, logits):  # block * batch * 2 * 1 * 1
        target = self.target_num * logits.size(0)
        logits = logits[:, :, 0, :, :].sum()
        return self.criterion(logits, torch.ones_like(logits) * target)


class DiversityEpochLoss(nn.Module):
    def __init__(self, total_blocks):
        super(DiversityEpochLoss, self).__init__()
        self.batch_count = 0.0
        self.total_blocks = total_blocks
        self.criterion = nn.L1Loss()
        self.sums = torch.zeros([self.total_blocks * 3], dtype=torch.float).cuda()

    def reset(self):
        self.sums = torch.zeros([self.total_blocks * 3], dtype=torch.float).cuda()
        self.batch_count = 0.0

    def forward(self, logits):
        self.sums = self.sums + logits
        self.batch_count = self.batch_count + 1.0
        target = self.sums / self.batch_count
        mean = target.mean()
        return self.criterion(logits, torch.ones_like(logits) * mean)


class DiversityBatchLoss(nn.Module):
    def __init__(self):
        super(FLOPSCELoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, logits):
        logits = logits.sum()
        return self.criterion(logits, torch.zeros_like(logits))


class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, logits):
        logits = logits.var(dim=1)
        return self.criterion(logits, torch.zeros_like(logits))


class UniformSkipLoss(nn.Module):
    def __init__(self, target, blocks):
        super(UniformSkipLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = torch.ones([blocks * 3], dtype=torch.float).cuda() * target / blocks

    def forward(self, logits):
        return self.criterion(logits, self.target)


class SmoothSkipLoss(nn.Module):
    def __init__(self, blocks):
        super(SmoothSkipLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.ones = torch.ones([blocks * 3], dtype=torch.float).cuda()
        self.mean = torch.tensor(0.5).cuda()

    def forward(self, logits):
        target = self.ones * self.mean
        self.mean = logits.mean().detach()
        return self.criterion(logits, target)


def GramMatrix(feature):
    # assert feature.shape[0] == 1
    assert feature.dim() == 3
    c, h, w = feature.shape
    feature = feature.view(c, h * w)
    gram = torch.mm(feature, feature.t())
    return gram


class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()

    def forward(self, fea1, fea2):
        gram1 = torch.stack([GramMatrix(fea) for fea in fea1])
        gram2 = torch.stack([GramMatrix(fea) for fea in fea2])
        return torch.mean(torch.nn.functional.l1_loss(gram1, gram2))


class ConDivLoss(nn.Module):
    def __init__(self, num_per_path, margin_intra, margin_inter, norm=2):
        super(ConDivLoss, self).__init__()
        self.norm = norm
        self.num_per_path = num_per_path
        self.margin_intra = margin_intra
        self.margin_inter = margin_inter

    def _get_center(self, path_fea):
        # import pdb; pdb.set_trace()
        batch_num, block_num, _, _, _ = path_fea.shape
        path_fea = path_fea.permute(1, 0, 2, 3, 4)
        path_fea = path_fea.view(block_num, batch_num // self.num_per_path, self.num_per_path, 2)
        center = torch.mean(path_fea, dim=2)
        return center.view(block_num, -1, 2).permute(1, 0, 2).contiguous().view(-1, block_num * 2)

    def _get_inter_loss(self, center):
        # center: num_sample, block * 2
        inter_term = 0
        # diff_all = []
        num_sample = center.shape[0]
        for i in range(num_sample):
            for j in range(i + 1, num_sample):
                diff = torch.norm(center[i, :] - center[j, :], self.norm)
                inter_term += torch.clamp(self.margin_inter - diff, min=0.0) ** 2
                # diff_all.append(diff)
        inter_term /= num_sample * (num_sample - 1) / 2
        return inter_term

    def _get_intra_loss(self, center, path_fea):
        # center : num_sample ,block * 2
        # path_fea : batch * block *  * 2 * 1 * 1
        num_sample = center.shape[0]
        # num_block = path_fea.shape[0]
        intra_term = 0
        for i in range(num_sample):
            center_i = center[i]
            # block * 2
            for j in range(i * self.num_per_path, (i + 1) * self.num_per_path):
                cur_fea = path_fea[j, :, :, :, :].contiguous().view(-1)
                diff = torch.norm(cur_fea - center_i, self.norm)
                intra_term += torch.clamp(diff - self.margin_intra, min=0.0) ** 2
        intra_term /= num_sample * self.num_per_path
        return intra_term

    def forward(self, path_fea):
        # fea :   batch* block * 2 * 1 * 1
        #print(path_fea.shape)
        assert path_fea.shape[0] % self.num_per_path == 0
        center = self._get_center(path_fea)
        inter_loss = self._get_inter_loss(center)
        intra_loss = self._get_intra_loss(center, path_fea)
        return inter_loss, intra_loss
