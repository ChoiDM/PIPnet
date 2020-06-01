import os
import sys
import torch
import torch.nn as nn
import numpy as np
from utils import Stopwatch

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False
    
def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

def ReplaceDenormals(net, thresh=1e-10, print_log=True):
    """Preventing learned parameters from being subnormal(denormal) numbers in floating-point representation
    """
    
    if print_log:
        print('Start to detect denormals in the trained-model, please wait for a while')
        total_denormals_count = 0
        total_normals_count = 0
        
    net = net.cpu()
    for name, param in net.named_parameters():
        
        if print_log:
            param_array = param.data.numpy()
            n_denormals = len(np.where((np.abs(param_array) < thresh) & (param_array != 0.0))[0])
            n_normals   = np.size(param_array) - n_denormals
            
            total_denormals_count += n_denormals
            total_normals_count   += n_normals
            
        param_denormed = torch.where(torch.abs(param) < thresh,
                                     torch.Tensor([0]),
                                     param)
        param.data.copy_(param_denormed.data)
    
    if print_log:
        total = total_normals_count + total_denormals_count
        print('All params: %d, normals: %d, denormals: %d, ratio: %f' % \
                (total, total_normals_count, total_denormals_count, total_denormals_count / total * 1.0))

def check_latency(net, c_in=3, s_size_h=256, s_size_w=256, repeat=500, bn_fold=True, replace_denormals=True):
    net.cpu()
    # net.mode = 'inference'

    if bn_fold:
        print("Batch Normalization Folding...")

        try:
            fuse_bn_recursively(net)
        except Exception as e:
            print("NOTE!!! Batch Normalization Failed. Error message is below\n", e)
    
    if replace_denormals:
        print("Replacing denormals...")
        ReplaceDenormals(net)

    torch.set_grad_enabled(False)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_num_threads(1)
    print('python version: %s' % sys.version)
    print('torch.__version__:%s' % torch.__version__)
    print('torch.backends.mkl.is_available(): %s' % torch.backends.mkl.is_available())
    print('torch.backends.openmp.is_available(): %s' % torch.backends.openmp.is_available())
    print(os.popen('conda list mkl').read())
    print('num_threads: %d' % torch.get_num_threads())

    batch_size = 1

    warm_start = 10
    repeat_count = 0

    timer = Stopwatch('latency', silance=True)

    elapsed = 0.
    for it in range(repeat + warm_start + 1):
        x = torch.rand(
            batch_size,
            c_in,
            s_size_h,
            s_size_w,
            requires_grad=False
        )

        with timer:
            out = net(x)
        
        if it > warm_start:
            elapsed += timer.latency
            repeat_count += 1
            
            if it % 10 == 0:
                print('trial: %d, latency %f' % (repeat_count, timer.latency))

    print('elapsed: %f' % (elapsed/repeat_count))