import os
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option(DATAROOT, USE_NSML=False, print_option=True):    
    p = argparse.ArgumentParser(description='')

    # Data Directory
    p.add_argument('--data_root', default='300W-LP/300W_train', type=str)
    p.add_argument('--csv_root', default='', type=str)
    p.add_argument('--DATASET', default='300W', type=str)
    p.add_argument('--test_dataset', default='full', type=str, help='full | common | challenge')
    p.add_argument('--offset_mode', default='point', type=str, help='grid | point (paper)')
    p.add_argument('--offset_dim', default=2, type=int, help='2(2D) or 1(1D)')
    p.add_argument('--offset_norm', default='False', type=str2bool, help='debugging (normalize offset values btw [0,1] or not)')
    
    # Data augmentation
    p.add_argument('--rot_factor', default=30, type=float)
    p.add_argument('--scale_factor', default=0., type=float)
    p.add_argument('--flip', default='True', type=str2bool)
    p.add_argument('--gaussian_blur', default='True', type=str2bool)
    p.add_argument('--occlusion', default='True', type=str2bool)
    p.add_argument('--trans_factor', default=0.05, type=float)

    # Input image
    p.add_argument('--in_res', default=256, type=int)
    p.add_argument('--mean', default=[0.485, 0.456, 0.406], type=list)
    p.add_argument('--std', default=[0.229, 0.224, 0.225], type=list)

    # Network
    p.add_argument('--backbone', default='resnet18', type=str, help='resnet18 | mobilev3s | mobilev3m | mobilev3l')
    p.add_argument('--backbone_resume', default='weights/resnet18_pretrained.pth', type=str)
    p.add_argument('--use_pretrained', default='True', type=str2bool)
    p.add_argument('--no_bias_decay', default='True', type=str2bool)
    p.add_argument('--width_mult', default=1.0, type=float)
    p.add_argument('--output_stride', default='32', type=str)
    p.add_argument('--mode', default='train', type=str, help='train | inference')
    p.add_argument('--head_version', default='paper', type=str)

    # Optimizer
    p.add_argument('--optim', default='Adam', type=str, help='RMSprop | SGD | Adam')
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--lr_warmup_epoch', default=-1, type=int)
    p.add_argument('--lr_decay_epoch', default='5,10,30,50', type=str, help="cosine | decay epochs with comma (ex - '20,40,60')")
    p.add_argument('--alpha', default=5, type=float, help='loss balancing')
    p.add_argument('--beta', default=3, type=float, help='loss balancing for translation training')
    p.add_argument('--eta_min_ratio', default=1e-3, type=float)
    p.add_argument('--momentum', default=0.9, type=float, help='momentum')
    p.add_argument('--wd', default=1e-5, type=float, help='weight decay')

    # Hyper-parameter
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--max_epoch', default=60, type=int)

    # Loss function
    p.add_argument('--hm_loss', default='MSE', type=str, help='AWing | Wing | MSE | L1 | L2 | BCE')
    p.add_argument('--off_loss', default='MSE', type=str, help='MSE | L1 | L2 | SL1 (Smooth L1 Loss)')
    p.add_argument('--weighted_map', default='True', type=str2bool, help='Apply weighted map to loss function (Only valid for AWing and Wing loss)')

    # Resume trained network
    p.add_argument('--resume', default='', type=str, help="pth file path")

    # Evaluation metrics option
    p.add_argument('--vis', action='store_true', help='visualize validation result in image')

    # Resource option
    p.add_argument('--workers', default=10, type=int, help='#data-loading worker-processes')
    p.add_argument('--gpu_id', default="0", type=str)

    # Output directory
    p.add_argument('--exp', default='dongmin/exp', type=str, help='checkpoint dir.')

    # Inference Option
    p.add_argument('--check_latency', action='store_true')
    p.add_argument('--bn_fold', action='store_true')
    p.add_argument('--replace_denormals', action='store_true')
    p.add_argument('--quantization', action='store_true')


    opt = p.parse_args()
    
    # Data Root
    opt.data_root = os.path.join(DATAROOT, opt.data_root)
    
    # NSML option
    if USE_NSML:
        from nsml import DATASET_PATH, NSML_NFS_OUTPUT
        print("DATASET PATH : %s - NSML NFS OUTPUT : %s" % (DATASET_PATH, NSML_NFS_OUTPUT))

        opt.data_root = os.path.join(DATASET_PATH[1], opt.data_root)
        opt.resume = os.path.join(NSML_NFS_OUTPUT, opt.resume)
        opt.exp = os.path.join(NSML_NFS_OUTPUT, opt.exp,
                            '%s_%s_width%s_os%s_%s_%dalpha_head_%s_lr%s_%s'%\
                            (opt.DATASET, opt.backbone, opt.width_mult, opt.output_stride, opt.offset_mode, opt.alpha, opt.head_version, opt.lr, opt.optim))
        opt.backbone_resume = os.path.join(NSML_NFS_OUTPUT, opt.backbone_resume)
        opt.csv_root = os.path.join(NSML_NFS_OUTPUT, opt.csv_root)

    # Make output directory
    os.system('mkdir -p %s' % opt.exp)

    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id

    opt.ngpu = len(opt.gpu_id.split(","))

    # Output Stide String Option to List
    if ',' in opt.output_stride:
        opt.output_stride = opt.output_stride.split(',')
        opt.output_stride = [int(os) for os in opt.output_stride]
    else:
        opt.output_stride = [int(opt.output_stride)]
    assert opt.output_stride in [[16], [32], [16, 32]]

    if ',' in opt.lr_decay_epoch:
        opt.lr_decay_epoch = opt.lr_decay_epoch.split(',')
        opt.lr_decay_epoch = [int(epoch) for epoch in opt.lr_decay_epoch]
    
    if print_option:
        print("\n==================================== Options ====================================\n")
    
        print('   Data root (%s DATASET): %s' % (opt.DATASET, opt.data_root))
        print()
        print('   lr: %f' % opt.lr)
        print('   gpu_id: %s' % opt.gpu_id)
        print('   resume: %s' % opt.resume)
        print('   batch size: %d' % opt.batch_size)
        print('   exp (checkpoint dir): %s' % opt.exp)
        # print('   loss function: %s (Weighted Map:%s)' % (opt.loss, opt.weighted_map))
        print()
        print('   pytorch version: %s' % torch.__version__)
        print("\n=================================================================================\n")

    return opt