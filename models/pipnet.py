import torch
import torch.nn as nn
import numpy as np

from models.resnet import resnet18, resnet50, resnet101
from models.resnet_v2 import resnet18d
from models.mobilenetv3 import MobileNetV3 as MobileNetV3_v1, MobileBottleneck as MobileBottleneck_v1
from models.mobilenetv3_new import MobileNetV3 as MobileNetV3_v2, MobileBottleneck as MobileBottleneck_v2
from models.CoordConv import CoordConv

class Head(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(Head,self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self,x):
        x = self.conv1x1(x)
        return x


class PIPNet(nn.Module):
    def __init__(self, backbone='resnet18', width_mult=1.0, n_landmarks=68, pretrained=False, opt=None):
        super(PIPNet, self).__init__()
        self.backbone_name = backbone
        self.mode = opt.mode
        self.opt = opt

        # Backbone
        if backbone == 'mobilev3s':
            self.backbone = MobileNetV3_v1(width_mult=width_mult, mode='small', output_stride=opt.output_stride, softgate=opt.softgate)

            in_channels_1 = self.make_divisible(int(48*width_mult))
            in_channels_2 = self.make_divisible(int(96*width_mult))

        elif backbone == 'resnet18':
            self.backbone = resnet18(width_mult=width_mult, output_stride=opt.output_stride)
            if width_mult == 1.0 and pretrained:
                self._load_pretrained_backbone()

            in_channels_1 = int(256*width_mult)
            in_channels_2 = int(512*width_mult)

        else:
            raise ValueError
        

        # Head
        if 16 in self.opt.output_stride:
            self.hm_head_1 = Head(in_channels_1, n_landmarks, output_stride=16)
            self.x_head_1 = Head(in_channels_1, n_landmarks, output_stride=16)
            self.y_head_1 = Head(in_channels_1, n_landmarks, output_stride=16)
 
        # OS = [16, 32]
        if 32 in self.opt.output_stride:
            self.hm_head_2 = Head(in_channels_2, n_landmarks, output_stride=32)
            self.x_head_2 = Head(in_channels_2, n_landmarks, output_stride=32)
            self.y_head_2 = Head(in_channels_2, n_landmarks, output_stride=32)

            nn.init.normal_(self.hm_head_2.conv1x1.weight, std=0.001)
            if self.hm_head_2.conv1x1.bias is not None:
                nn.init.constant_(self.hm_head_2.conv1x1.bias, 0)

            nn.init.normal_(self.x_head_2.conv1x1.weight, std=0.001)
            if self.x_head_2.conv1x1.bias is not None:
                nn.init.constant_(self.x_head_2.conv1x1.bias, 0)

            nn.init.normal_(self.y_head_2.conv1x1.weight, std=0.001)
            if self.y_head_2.conv1x1.bias is not None:
                nn.init.constant_(self.y_head_2.conv1x1.bias, 0)


    def make_divisible(self, x, divisible_by=8):
        return int(np.ceil(x * 1. / divisible_by) * divisible_by)


    def forward(self, x):
        # 8x8 output
        if self.opt.output_stride == [32]:
            _, x = self.backbone(x)

            hm = self.hm_head_2(x)
            x_off = self.x_head_2(x)
            y_off = self.y_head_2(x)

            return [(hm, x_off, y_off)]

        # 16x16 output
        elif self.opt.output_stride == [16]:
            x = self.backbone(x)

            hm = self.hm_head_1(x)
            x_off = self.x_head_1(x)
            y_off = self.y_head_1(x)

            return [(hm, x_off, y_off)]

        # 8x8 output & 16x16 output (only for supervision training)
        elif self.opt.output_stride == [16, 32]:
            x1, x2 = self.backbone(x)

            hm2 = self.hm_head_2(x2)
            x_off2 = self.x_head_2(x2)
            y_off2 = self.y_head_2(x2)
            
            if self.mode == 'train':
                hm1 = self.hm_head_1(x1)
                x_off1 = self.x_head_1(x1)
                y_off1 = self.y_head_1(x1)

                return [(hm1, x_off1, y_off1), (hm2, x_off2, y_off2)]
            
            else:
                return [(hm2, x_off2, y_off2)]
        
            
    def _load_pretrained_backbone(self):
        pth_path = self.opt.backbone_resume

        pretrained_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        model_dict = self.backbone.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict) 
        self.backbone.load_state_dict(model_dict)

        print(">>> Pre-trained backbone loaded")