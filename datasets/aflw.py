# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

from utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AFLW(data.Dataset):
    """AFLW
    """
    def __init__(self, opt, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = os.path.join(opt.csv_root, 'data/aflw/face_landmarks_aflw_train.csv')
        else:
            self.csv_file = os.path.join(opt.csv_root, 'data/aflw/face_landmarks_aflw_test.csv')

        self.is_train = is_train
        self.transform = transform
        self.data_root = opt.data_root
        self.offset_mode = opt.offset_mode
        self.offset_dim = opt.offset_dim
        self.offset_norm = opt.offset_norm

        self.input_size = (opt.in_res, opt.in_res)

        if isinstance(opt.output_stride, list):
            self.output_size = [(opt.in_res//os, opt.in_res//os) for os in opt.output_stride]
        else:
            self.output_size = [(opt.in_res//opt.output_stride, opt.in_res//opt.output_stride)]

        self.scale_factor = opt.scale_factor
        self.rot_factor = opt.rot_factor
        self.flip = opt.flip
        self.gaussian_blur = opt.gaussian_blur

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)
        # self.WFLW_98_TO_300W_68_PTS()

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
                
                if self.offset_norm:
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
                
                if self.offset_norm:
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
                
                if self.offset_norm:
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
                
                if self.offset_norm:
                    xx_channel = xx_channel / output_size[0]
                    yy_channel = yy_channel / output_size[1]

            return xx_channel, yy_channel

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        image_path = image_path.replace('flickr/', '')
        
        image_name = self.landmarks_frame.iloc[idx, 0]
        scale = self.landmarks_frame.iloc[idx, 1]
        box_size = self.landmarks_frame.iloc[idx, 2]

        center_w = self.landmarks_frame.iloc[idx, 3]
        center_h = self.landmarks_frame.iloc[idx, 4]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 5:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='AFLW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)
        if random.random() <= 0.3 and self.gaussian_blur:
            radius = random.choice([1, 3, 5])
            img = cv2.GaussianBlur(img, (radius, radius), sigmaX=1.0)

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

            xx_channel, yy_channel = self._generate_offset(nparts, tpts, output_size, self.offset_mode, self.offset_dim, )
            offset = [torch.Tensor(xx_channel), torch.Tensor(yy_channel)]

            targets.append(torch.Tensor(target))
            offsets.append(offset)

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'image_name':image_name, 'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        return img, targets, offsets, meta


if __name__ == '__main__':
    pass