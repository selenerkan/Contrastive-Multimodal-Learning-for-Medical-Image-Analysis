import torch
from torch import nn
import numpy as np
import random
from pytorch_lightning import seed_everything
from ham_settings import SEED
import os
from pytorch_lightning.core.module import LightningModule


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()

        seed_everything(SEED, workers=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
            print('centers', self.centers)
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))
            print('centers', self.centers)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# ---------------------------------------------------------------------------------------------------------------------------
# import torch


# def compute_center_loss(features, centers, targets):

#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'

#     features = features.to(device)
#     centers = centers.to(device)
#     targets = targets.to(device)

#     features = features.view(features.size(0), -1)
#     target_centers = centers[targets]
#     criterion = torch.nn.MSELoss()
#     center_loss = criterion(features, target_centers)
#     return center_loss


# def get_center_delta(features, centers, targets, alpha):

#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'

#     features = features.to(device)
#     centers = centers.to(device)
#     targets = targets.to(device)
#     alpha = alpha.to(device)

#     # implementation equation (4) in the center-loss paper
#     features = features.view(features.size(0), -1)
#     targets, indices = torch.sort(targets)
#     target_centers = centers[targets]
#     features = features[indices]

#     delta_centers = target_centers - features
#     uni_targets, indices = torch.unique(
#         targets, sorted=True, return_inverse=True)

#     uni_targets = uni_targets.to(device)
#     indices = indices.to(device)

#     delta_centers = torch.zeros(
#         uni_targets.size(0), delta_centers.size(1)
#     ).to(device).index_add_(0, indices, delta_centers)

#     targets_repeat_num = uni_targets.size()[0]
#     uni_targets_repeat_num = targets.size()[0]
#     targets_repeat = targets.repeat(
#         targets_repeat_num).view(targets_repeat_num, -1)
#     uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
#         1, uni_targets_repeat_num)
#     same_class_feature_count = torch.sum(
#         targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

#     delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
#     result = torch.zeros_like(centers)
#     result[uni_targets, :] = delta_centers
#     return result
