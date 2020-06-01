import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', softgate=False):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.softgate = softgate
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

        if self.use_res_connect and self.softgate:
            self.alpha = torch.nn.Parameter(torch.zeros(1,oup,1,1), requires_grad=True)

    def forward(self, x):
        if self.use_res_connect:
            if self.softgate:
                return self.alpha*x + self.conv(x)

            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, mode='small', width_mult=1.0, output_stride=[16,32], softgate=False):
        super(MobileNetV3, self).__init__()
        self.output_stride = output_stride

        input_channel = 16
        last_channel = 1280

        if mode == 'small':
            # refer to Table 2 in paper
            mobile_setting_1 = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [3, 96,  40,  True,  'HS', 2],
                [3, 240, 40,  True,  'HS', 1],
                [3, 240, 40,  True,  'HS', 1],
                [3, 120, 48,  True,  'HS', 1],
                [3, 144, 48,  True,  'HS', 1]
            ]
            
            mobile_setting_2 = [
                # k, exp, c,  se,     nl,  s,
                [3, 288, 96,  True,  'HS', 2],
                [3, 576, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1]
            ]
        
        elif mode == 'small_relu':
            # refer to Table 2 in paper
            mobile_setting_1 = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [3, 96,  40,  True,  'RE', 2],
                [3, 240, 40,  True,  'RE', 1],
                [3, 240, 40,  True,  'RE', 1],
                [3, 120, 48,  True,  'RE', 1],
                [3, 144, 48,  True,  'RE', 1]
            ]
            
            mobile_setting_2 = [
                # k, exp, c,  se,     nl,  s,
                [3, 288//2, 96,  True,  'RE', 2],
                [3, 576//2, 96,  True,  'RE', 1],
                [3, 576//2, 96,  True,  'RE', 1]
            ]


        elif mode == 'large':
            # refer to Table 1 in paper
            mobile_setting_1 = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [3, 72,  40,  True,  'RE', 2],
                [3, 120, 40,  True,  'RE', 1],
                [3, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1]
            ]

            mobile_setting_2 = [
                # k, exp, c,  se,     nl,  s,
                [3, 672, 160, True,  'HS', 2],
                [3, 960, 160, True,  'HS', 1],
                [3, 960, 160, True,  'HS', 1]
            ]

        elif mode == 'medium':
            # refer to Table 2 in paper
            mobile_setting_1 = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [3, 96,  40,  True,  'HS', 2],
                [3, 96,  40,  True,  'HS', 1],
                [3, 240, 40,  True,  'HS', 1],
                [3, 240, 40,  True,  'HS', 1],
                [3, 120, 44,  True,  'HS', 1],
                [3, 120, 48,  True,  'HS', 1],
                [3, 144, 48,  True,  'HS', 1],
                [3, 144, 48,  True,  'HS', 1]
            ]

            mobile_setting_2 = [
                # k, exp, c,  se,     nl,  s,
                [3, 288, 72,  True,  'HS', 2],
                [3, 288, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1],
                [3, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.conv1 = conv_bn(3, input_channel, 2, nlin_layer=Hswish)

        # building mobile blocks
        self.blocks1 = []
        for k, exp, c, se, nl, s in mobile_setting_1:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.blocks1.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, softgate))
            input_channel = output_channel
        self.blocks1 = nn.Sequential(*self.blocks1)

        if 32 in self.output_stride:
            self.blocks2 = []
            for k, exp, c, se, nl, s in mobile_setting_2:
                output_channel = make_divisible(c * width_mult)
                exp_channel = make_divisible(exp * width_mult)
                self.blocks2.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, softgate))
                input_channel = output_channel
            self.blocks2 = nn.Sequential(*self.blocks2)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.blocks1(x)

        if 32 in self.output_stride:
            x2 = self.blocks2(x1)
            return x1, x2
        
        else:
            return x1

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('weights/mobilenetv3_small_67.4.pth.tar', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == '__main__':
    net = mobilenetv3()
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3, 224, 224)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile
    flops, params = profile(net, input_size=input_size)
    # print(flops)
    # print(params)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = net(x)


