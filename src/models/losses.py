import torch
import torch.nn as nn
import torch.nn.functional as F


class DMILoss(nn.Module):

    def __init__(self, num_classes: int):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes)
        joint_dist = output.transpose(0, 1).float() @ target_one_hot.float()
        joint_dist /= target.size(0)
        return -1.0 * torch.log(torch.abs(torch.det(joint_dist.float())) + 1e-4)
