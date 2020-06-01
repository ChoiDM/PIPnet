import os
import cv2
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import glob

from models.pipnet import PIPNet
from models.latency import check_latency, fuse_bn_recursively, ReplaceDenormals

def create_model(opt):
  if opt.DATASET=='300W':
    n_landmarks = 68
  elif opt.DATASET=='WFLW':
    n_landmarks = 98
  else:
    raise ValueError('Check DATASET argument name again.')

  opt.n_landmarks = n_landmarks
  
  net = PIPNet(opt=opt, backbone=opt.backbone, n_landmarks=n_landmarks,
               width_mult=opt.width_mult, pretrained=opt.use_pretrained)

  if opt.resume:
    if os.path.isfile(opt.resume):
      pretrained_dict = torch.load(opt.resume, map_location=torch.device('cpu'))
      model_dict = net.state_dict()

      match_cnt = 0
      mismatch_cnt = 0
      pretrained_dict_matched = dict()
      for k, v in pretrained_dict.items():
          if k in model_dict and v.size() == model_dict[k].size():
              pretrained_dict_matched[k] = v
              match_cnt += 1
          else:
              mismatch_cnt += 1
              
      model_dict.update(pretrained_dict_matched) 
      net.load_state_dict(model_dict)

      print("Successfully loaded weights from %s" % opt.resume)
      print("# of matched weights : %d / # of mismatched weigths : %d" % (match_cnt, mismatch_cnt))
    else:
      print("=> no checkpoint found at '{}'".format(opt.resume))

  if opt.ngpu > 1:
    net.cuda()
    net = torch.nn.DataParallel(net)

  if opt.bn_fold:
    fuse_bn_recursively(net)

  if opt.replace_denormals:
    ReplaceDenormals(net)

  if opt.check_latency:
    check_latency(net, 3, opt.in_res, opt.in_res, repeat=500, replace_denormals=True, bn_fold=True)

  return net
