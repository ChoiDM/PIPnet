import os
import cv2
import numpy as np

import torch

from utils.metrics import get_preds

def visualize(dataset_val, net, epoch, opt):
  save_dir = os.path.join(opt.exp, 'vis', 'epoch_%04d'%epoch)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  global_iter = 0
  for imgs, heatmaps, _ in dataset_val:
    global_iter = visualize_batch(imgs, heatmaps, net, save_dir, opt, global_iter)
    

def visualize_batch(imgs, heatmaps, offsets, res, net, save_dir, opt, global_iter):
  with torch.no_grad():
    pred_heatmaps, pred_x, pred_y = net(imgs)[0]
  pred_offset = pred_x, pred_y

  score_map = pred_heatmaps.data.cpu()
  pred_pts = get_preds(score_map, pred_offset, res).numpy()
  true_pts = get_preds(heatmaps, offsets, res).numpy()

  for img, pred_pt, true_pt in zip(imgs.cpu().data.numpy(), pred_pts, true_pts):
    img = np.moveaxis(img, 0, -1)
    img = ((img * opt.std + opt.mean) * 255).astype(np.uint8).copy()

    for pt in pred_pt:
      x,y = (pt*opt.output_stride[-1]).astype(int)
      cv2.circle(img, (x,y), radius=2, color=(255,0,0), thickness=-1)
    for pt in true_pt:
      x,y = (pt*opt.output_stride[-1]).astype(int)
      cv2.circle(img, (x,y), radius=2, color=(0,0,255), thickness=-1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, 'iter_%04d.png'%global_iter), img)
    global_iter += 1

  return global_iter