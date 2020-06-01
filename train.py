from __future__ import print_function

import os
import random
import importlib
import matplotlib
import numpy as np
matplotlib.use('agg')

try:
  nsml = importlib.import_module('nsml')
  DATASET_PATH = nsml.DATASET_PATH
  NSML_NFS_OUTPUT = nsml.NSML_NFS_OUTPUT
  USE_NSML = True
  print('\nThis script will be ran on the NSML')
except ImportError as e:
  nsml = None
  USE_NSML = False
  print('\nThis script will be ran on the local machine')

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

from models import create_model
from models.core import train, evaluate

from utils import get_optimizer, lr_update, nsml_bind_model
from utils.loss import get_loss_function
from datasets import get_dataloader

from options import parse_option

import warnings
warnings.filterwarnings('ignore')


# Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Option
DATAROOT = '/home/taey16/storage' if not USE_NSML else os.path.join(DATASET_PATH[1], 'taey16', 'storage')
opt = parse_option(DATAROOT, USE_NSML=USE_NSML, print_option=False)

# Data Loader
dataset_trn, dataset_val = get_dataloader(opt)

# Loading model
net = create_model(opt)

# Loss Function
hm_criterion, off_criterion = get_loss_function(opt)

# Optimizer
optimizer = get_optimizer(net, opt)
scheduler = CosineAnnealingLR(optimizer, eta_min=opt.lr*opt.eta_min_ratio, T_max=(opt.max_epoch - opt.lr_warmup_epoch))

# Initial Best Score
global_iter, best_nme, best_epoch = [0, 10000, 0]

#NOTE: main loop for training
if __name__ == "__main__":

  if USE_NSML:
    scope = locals()
    nsml_bind_model(nsml, scope, 0, net, optimizer)
  else:
    scope = None

  for epoch in range(opt.start_epoch, opt.max_epoch):
    # Train
    global_iter = train(dataset_trn, net, hm_criterion, off_criterion, optimizer, epoch, global_iter, opt, nsml, scope)

    # Evaluate
    best_nme, best_epoch = evaluate(dataset_val, net, hm_criterion, off_criterion, optimizer, epoch, opt, best_nme, best_epoch, global_iter, nsml, scope)

    lr_update(epoch, opt, optimizer, scheduler)

  print('Training done')