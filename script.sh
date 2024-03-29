#!/bin/bash

#Train ANN
python ann.py --dataset CIFAR10 --batch_size 128 --architecture RESNET12 --learning_rate 1e-3 --epochs 300 --optimizer Adam --dropout 0.2 --devices 2 --log

#Train SNN
#python snn.py --dataset CIFAR10 --batch_size 64 --architecture RESNET12 --learning_rate 1e-4 --optimizer 'Adam' --epochs 300 --timesteps 5 --scaling_factor 0.2 --weight_decay 0 --dropout 0.2 --train_acc_batches 500 --devices 2 --default_threshold 0.4 --pretrained_ann './trained_models/ann/ann_resnet12_cifar10_Fri Sep 10 18:02:00 2021.pth' --dont_save --log #--individual_thresh #--dont_save #--test_only --test_acc_every_batch 