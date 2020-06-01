import numpy as np
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()

        self.omega = omega
        self.epsilon = epsilon

        self.weighted_map = weighted_map

    def forward(self, prediction, target):
        C = (self.omega - self.omega * np.log(1.0+self.omega/self.epsilon))

        diff_abs = torch.abs(prediction-target)
        loss = torch.where(diff_abs < self.omega,
                           self.omega * torch.log(1.0+diff_abs/self.epsilon),
                           diff_abs - C
        )


class AWingLoss(nn.Module):
    def __init__(self, omega=14.0, epsilon=1.0, theta=0.5, alpha=2.1, weighted_map=True, add_boundary=False):
        super(AWingLoss, self).__init__()

        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        self.alpha = alpha

        self.weighted_map = weighted_map
        self.add_boundary = add_boundary

    def forward(self, prediction, target):
        A = self.omega * (1.0/(1.0+torch.pow(self.theta/self.epsilon, self.alpha-target))) * (self.alpha-target) * torch.pow(self.theta/self.epsilon, self.alpha-target-1.0) * (1.0/self.epsilon)
        C = (self.theta*A - self.omega*torch.log(1.0+torch.pow(self.theta/self.epsilon, self.alpha-target)))

        diff_abs = torch.abs(prediction-target)
        loss = torch.where(diff_abs < self.theta,
                           self.omega * torch.log(1.0+torch.pow(diff_abs/self.epsilon, self.alpha-target)),
                           A * diff_abs - C
        )

        if self.weighted_map:
            loss *= self.generate_loss_map_mask(target, self.add_boundary)

        return loss.mean()
    
    def generate_loss_map_mask(self, target, W=10.0, k_size=3, threshold=0.2, add_boundary=False):
        target_array = target.cpu().numpy()
        mask = np.zeros_like(target_array)

        for batch in range(mask.shape[0]):
            for loc in range(mask.shape[1]-1 if add_boundary else mask.shape[1]):
                H_d = ndimage.grey_dilation(target_array[batch, loc], size=(k_size, k_size))
                mask[batch, loc, H_d > threshold] = W
        
        return torch.Tensor(mask+1).to(target.device)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

def get_loss_function(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.hm_loss == 'BCE':
        hm_criterion = nn.BCEWithLogitsLoss(reduction='sum')

    elif opt.hm_loss == 'MSE':
        hm_criterion = nn.MSELoss(reduction='mean')

    elif opt.hm_loss == 'L1':
        hm_criterion = nn.L1Loss(reduction='mean')

    elif opt.hm_loss == 'L2':
        hm_criterion = nn.MSELoss(reduction='sum')

    elif opt.hm_loss == 'AWing':
        hm_criterion = AWingLoss(weighted_map=opt.weighted_map, add_boundary=False)
    
    elif opt.hm_loss == 'Wing':
        hm_criterion = WingLoss(weighted_map=opt.weighted_map, add_boundary=False)

    else:
        raise ValueError("Only 'BCE', 'MSE', 'AWing', 'Wing' loss function are supported now.")

    if opt.off_loss == 'SL1':
        off_criterion = nn.SmoothL1Loss(reduction='mean')

    elif opt.off_loss == 'L1':
        off_criterion = nn.L1Loss(reduction='mean')

    elif opt.off_loss == 'L2':
        off_criterion = nn.MSELoss(reduction='sum')

    elif opt.off_loss == 'MSE':
        off_criterion = nn.MSELoss(reduction='mean')

    return hm_criterion, off_criterion