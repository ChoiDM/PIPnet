# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
from .face300w import Face300W
from .wflw_68pts import WFLW
from .aflw_68pts import AFLW
from torch.utils.data import DataLoader


def get_dataset(opt):
    if opt.DATASET == '300W':
        return Face300W
    elif opt.DATASET == 'WFLW':
        return WFLW
    elif opt.DATASET == 'AFLW':
        return AFLW
    else:
        raise NotImplemented()

def get_dataloader(opt):
    dataset = get_dataset(opt)

    train_dataloader = DataLoader(dataset(opt, is_train=True),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    valid_dataloader = DataLoader(dataset(opt, is_train=False),
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.workers)
    
    return train_dataloader, valid_dataloader


def get_joint_dataloader(DATAROOT, opt):
    opt.data_root = os.path.join(DATAROOT, '300W-LP', '300W_train')
    w300_train_dataloader = DataLoader(Face300W(opt, is_train=True),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    w300_valid_dataloader = DataLoader(Face300W(opt, is_train=False),
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.workers)
    
    opt.data_root = os.path.join(DATAROOT, 'WFLW', 'WFLW_images')
    wflw_train_dataloader = DataLoader(WFLW(opt, is_train=True),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    opt.data_root = os.path.join(DATAROOT, 'AFLW')
    aflw_train_dataloader = DataLoader(AFLW(opt, is_train=True),
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    return w300_train_dataloader, w300_valid_dataloader, wflw_train_dataloader, aflw_train_dataloader