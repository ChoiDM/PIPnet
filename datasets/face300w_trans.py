# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data

from utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class Face300W(data.Dataset):

    def __init__(self, opt, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = 'data/300w/face_landmarks_300w_train.csv'
        else:
            if opt.test_dataset == 'full':
                print("Full 300W Validation Dataset")
                self.csv_file = 'data/300w/face_landmarks_300w_valid.csv'
            elif opt.test_dataset == 'common':
                print("Common 300W Validation Dataset")
                self.csv_file = 'data/300w/face_landmarks_300w_valid_common.csv'
            elif opt.test_dataset == 'challenge':
                print("Challenge 300W Validation Dataset")
                self.csv_file = 'data/300w/face_landmarks_300w_valid_challenge.csv'

        self.is_train = is_train
        self.transform = transform
        self.data_root = opt.data_root
        self.offset_mode = opt.offset_mode
        self.offset_dim = opt.offset_dim

        self.input_size = (opt.in_res, opt.in_res)

        if isinstance(opt.output_stride, list):
            self.output_size = [(opt.in_res//os, opt.in_res//os) for os in opt.output_stride]
        else:
            self.output_size = [(opt.in_res//opt.output_stride, opt.in_res//opt.output_stride)]

        self.trans_factor = opt.trans_factor
        self.n_trans = 8

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array(opt.mean, np.float32)
        self.std = np.array(opt.std, np.float32)

    def __len__(self):
        return len(self.landmarks_frame)
    
    def _generate_offset(self, nparts, tpts, output_size, mode='grid', dim=2):
        if dim == 2:
            if mode == 'grid':
                xx_ones = np.ones([nparts, output_size[0]], dtype=np.int32)
                xx_ones = np.expand_dims(xx_ones, -1)

                xx_range = np.expand_dims(np.arange(output_size[1]), 0)
                xx_range = np.expand_dims(xx_range, 1)

                xx_channel = np.matmul(xx_ones, xx_range)
            
                yy_ones = np.ones([nparts, output_size[0]], dtype=np.int32)
                yy_ones = np.expand_dims(yy_ones, 1)

                yy_range = np.expand_dims(np.arange(output_size[1]), 0)
                yy_range = np.expand_dims(yy_range, -1)

                yy_channel = np.matmul(yy_range, yy_ones)

                xx_channel = (-xx_channel + tpts[:,0][:, None, None])
                yy_channel = (-yy_channel + tpts[:,1][:, None, None])
                
                xx_channel = xx_channel / output_size[0]
                yy_channel = yy_channel / output_size[1]
            
            elif mode == 'point':
                xx_channel = np.zeros([nparts, output_size[0], output_size[1]], dtype=np.float32)
                yy_channel = np.zeros([nparts, output_size[0], output_size[1]], dtype=np.float32)

                for i in range(nparts):
                    if tpts[i, 1] > 0:
                        x, y = tpts[i].astype(int)

                        if 0 <= y < output_size[0] and 0 <= x < output_size[1]:
                            x_off, y_off = tpts[i] - tpts[i].astype(int)
                            xx_channel[i, y, x] = x_off
                            yy_channel[i, y, x] = y_off
                
                xx_channel = xx_channel / output_size[0]
                yy_channel = yy_channel / output_size[1]

            return xx_channel, yy_channel
        
        elif dim == 1:
            if mode == 'grid':
                xx_channel = np.expand_dims(np.arange(output_size[1]), 0)
                xx_channel = np.expand_dims(xx_channel, 1)
                xx_channel = np.repeat(xx_channel, repeats=nparts, axis=0)

                yy_channel = np.expand_dims(np.arange(output_size[1]), 0)
                yy_channel = np.expand_dims(yy_channel, -1)
                yy_channel = np.repeat(yy_channel, repeats=nparts, axis=0)

                xx_channel = (-xx_channel + tpts[:,0][:, None, None])
                yy_channel = (-yy_channel + tpts[:,1][:, None, None])
                
                xx_channel = xx_channel / output_size[0]
                yy_channel = yy_channel / output_size[1]
            
            elif mode == 'point':
                xx_channel = np.zeros([nparts, 1, output_size[1]], dtype=np.float32)
                yy_channel = np.zeros([nparts, output_size[0], 1], dtype=np.float32)

                for i in range(nparts):
                    if tpts[i, 1] > 0:
                        x, y = tpts[i].astype(int)

                        if 0 <= y < output_size[0] and 0 <= x < output_size[1]:
                            x_off, y_off = tpts[i] - tpts[i].astype(int)
                            xx_channel[i, 0, y] = x_off
                            yy_channel[i, x, 0] = y_off
                
                xx_channel = xx_channel / output_size[0]
                yy_channel = yy_channel / output_size[1]

            return xx_channel, yy_channel

    def __getitem__(self, idx):
        if self.landmarks_frame.iloc[idx, 0].find('/ibug/image_092_01.jpg'):
            self.landmarks_frame.iloc[idx, 0] = self.landmarks_frame.iloc[idx, 0].replace('image_092_01.jpg', 'image_092 _01.jpg')

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])

        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        img = crop(img, center, scale, self.input_size, rot=r)

        trans_imgs = []
        trans_factors = []
        for _ in range(self.n_trans):
            x_factor = random.uniform(-self.trans_factor, self.trans_factor)
            y_factor = random.uniform(-self.trans_factor, self.trans_factor)

            M = np.float32([[1,0,self.input_size[0]*x_factor],[0,1,self.input_size[1]*y_factor]])
            trans_img = cv2.warpAffine(img, M, self.input_size)

            trans_img = trans_img.astype(np.float32)
            trans_img = (trans_img/255.0 - self.mean) / self.std
            trans_img = trans_img.transpose([2, 0, 1])

            trans_imgs.append(trans_img)
            trans_factors.append([x_factor, y_factor])
        
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        targets = []
        offsets = []
        for output_size in self.output_size:
            target = np.zeros((nparts, output_size[0], output_size[1]))
            tpts = pts.copy()

            for i in range(nparts):
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = transform_pixel(tpts[i, 0:2], center,
                                                scale, output_size, rot=r)
                    x, y = tpts[i].astype(int)

                    if 0 <= y < output_size[0] and 0 <= x < output_size[1]:
                        target[i, y, x] = 1

            xx_channel, yy_channel = self._generate_offset(nparts, tpts, output_size, self.offset_mode, self.offset_dim)
            offset = [torch.Tensor(xx_channel), torch.Tensor(yy_channel)]

            targets.append(torch.Tensor(target))
            offsets.append(offset)

        trans_imgs = torch.Tensor(trans_imgs)
        trans_factors = torch.Tensor(trans_factors)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, trans_imgs, trans_factors, targets, offsets, meta


if __name__ == '__main__':
    pass