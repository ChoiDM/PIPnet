import numpy as np
import torch

import math
from utils.transforms import transform_preds

def get_preds(scores, offset=None, res=(8,8), offset_norm=True):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'

    if offset is None:
        offset = [torch.zeros_like(scores).to(scores.device), torch.zeros_like(scores).to(scores.device)]

    x_offset, y_offset = offset
    _, _, w, h = x_offset.shape

    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)

    x_offset = x_offset.view(scores.size(0), scores.size(1), -1).cpu()
    x_mask = torch.tensor(np.eye(x_offset.size(-1))[idx[:,:,0]].astype(bool))
    x_offset = x_offset[x_mask].view(scores.size(0), scores.size(1))

    y_offset = y_offset.view(scores.size(0), scores.size(1), -1).cpu()
    y_mask = torch.tensor(np.eye(y_offset.size(-1))[idx[:,:,0]].astype(bool))
    y_offset = y_offset[y_mask].view(scores.size(0), scores.size(1))

    if offset_norm:
        x_offset *= w
        y_offset *= h

    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3) + x_offset
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3)) + y_offset
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def compute_nme(preds, meta):
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')

        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def decode_preds(output, offset, center, scale, res, offset_norm=True):
    coords = get_preds(output, offset, res, offset_norm)  # float type
    coords = coords.cpu()
    # pose-processing
    # for n in range(coords.size(0)):
    #     for p in range(coords.size(1)):
    #         hm = output[n][p]
    #         px = int(math.floor(coords[n][p][0]))
    #         py = int(math.floor(coords[n][p][1]))
    #         if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
    #             diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
    #             coords[n][p] += diff.sign() * .25
    #             # coords[n][p] += diff.sign()
    # # coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds-1


def get_preds_2(scores, offset, res, offset_norm=True):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    scores = scores.cpu()

    res_x, res_y = res
    x_offset, y_offset = [os.cpu() for os in offset]
    b, c, h, w = scores.shape

    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)
    
    col_idx = (idx % w)[:,:,0]
    col_mask = torch.eye(w)[col_idx].byte()
    x_offset = x_offset.transpose(2, 3)
    x_offset = x_offset[col_mask].view(b, c)
    
    if offset_norm:
        x_offset *= w
        y_offset *= h

    row_idx = (idx // h)[:,:,0]
    row_mask = torch.eye(h)[row_idx].byte()
    y_offset = y_offset[row_mask].view(b, c)

    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3) + x_offset
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3)) + y_offset
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds
    
def decode_preds_2(output, offset, center, scale, res, offset_norm=True):
    coords = get_preds_2(output, offset, res, offset_norm)  # float type
    coords = coords.cpu()
    # pose-processing
    # for n in range(coords.size(0)):
    #     for p in range(coords.size(1)):
    #         hm = output[n][p]
    #         px = int(math.floor(coords[n][p][0]))
    #         py = int(math.floor(coords[n][p][1]))
    #         if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
    #             diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
    #             coords[n][p] += diff.sign() * .25
    #             # coords[n][p] += diff.sign()
    # # coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds-1