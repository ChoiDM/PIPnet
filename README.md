# PIPNet
Implementation of [Pixel-In-Pixel Net](https://arxiv.org/abs/2003.03771)

## NME Performance

|  Backbone   | common  | challenge | full | Latency (CPU) | Params(M) | MACs(M) |
| :---------: | :-----: | :-------: | :--: | :-----------: | :-------: | :-----: |
| Mobilev3-16 |  4.12   |   7.80    | 4.84 |     7.79      |   0.173   |  48.35  |
| Mobilev3-32 |  3.75   |   6.80    | 4.35 |     10.12     |   0.683   |  56.92  |

- Evaluated on 300W
- CPU latency (ms) is tested on Macbook Pro with 1.4 GHz quad core Intel Core i5 and pytorch 1.5.0

## Quick Start
```
# Training on Local
python train.py --backbone mobilev3s --width_mult 1.0 --lr 1e-4 --batch_size 64 --mode train --exp output_dir 

# Training on NSML
nsml run -e train.py -d prod-nfs -d prod-nfs2 --nfs-output -a '--backbone mobilev3s --width_mult 1.0 --lr 1e-4 --batch_size 64 --mode train --exp output_dir' -g 1 -c 14 --shm-size 5G --gpu-driver-version 418.39

# Test
python test.py --backbone mobilev3s --width_mult 1.0 --resume trained_weights.pth --test_dataset full --mode inference
```

## Training Details
- Learning Rate = Initial 2.5e-5, Warmup 2.5e-4 (5 epoch), Cosine Annealing to 2.5e-7 (800 epochs, w/ three restart)
- Loss = Score map : L2, Offset : L1 (balacing factor alpha : 5)
- Optimizer = RMSprop (momentum : 0.9, weight decay : 1e-5)
- Batch size = 64
- Resolution = Input : 256x256, Output : 8x8 (Output stride 32)


## Result (Mobilev3-32)
![](figure/001_1D_ex.gif)
![](figure/037_1D_ex.gif)
![](figure/114_1D_ex.gif)
![](figure/512_1D_ex.gif)

## Light Version Result (Mobilev3-16)
![](figure/001_1D_light_ex.gif)
![](figure/037_1D_light_ex.gif)
![](figure/114_1D_light_ex.gif)
![](figure/512_1D_light_ex.gif)

## Study Summary (20200520)
[[20200520] PIPnet Progress.pdf](https://oss.navercorp.com/clova-face/face_keypoint_train/blob/baseline/PIPNet/%5B20200520%5D%20PIPnet%20Progress.pdf)