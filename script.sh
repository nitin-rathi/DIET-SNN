#!/bin/bash

#Train ANN
python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 300 --optimizer Adam --dropout 0.2 --devices 2 --dont_save --log

#python snn.py --dataset CIFAR100 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --optimizer 'Adam' --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 10 --scaling_factor 1.0 --weight_decay 0 --dropout 0.2 --train_acc_batches 500 --devices 0 --default_threshold 0.4 --pretrained_snn './trained_models/snn/snn_vgg16_cifar100_5_2020-10-20 15:03:08.219796.pth' --log & #--individual_thresh #--dont_save #--test_only --test_acc_every_batch 

#python snn.py --dataset CIFAR10 --batch_size 128 --architecture RESNET12 --learning_rate 1e-3 --optimizer 'SGD' --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 5 --scaling_factor 1.0 --weight_decay 0 --dropout 0.2 --train_acc_batches 200 --devices 2 --default_threshold 0.4 --dont_save #--test_only --test_acc_every_batch #--log &

#CIFAR10
#python snn.py --dataset CIFAR100 --batch_size 512 --architecture VGG16 --learning_rate 1e-4 --epochs 2 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 5 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --momentum 0.90 --dropout 0.1 --train_acc_batches 500 --devices 2 --pretrained_snn './trained_models/snn/snn_vgg16_cifar100_5_2020-10-20 15:03:08.219796.pth' --test_only --test_acc_every_batch --dont_save



#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 12 --leak 1.0 --scaling_factor 0.6 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 50 --devices 1 --default_threshold 0.5 --pretrained_ann './trained_models/ann/ann_vgg16_cifar10.pth' --dont_save #--log & #--test_only --test_acc_every_batch

#python snn.py --dataset CIFAR100 --batch_size 64 --architecture RESNET20 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 10 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --amsgrad True --dropout 0.2 --train_acc_batches 500 --devices 2 --default_threshold 0.55 --pretrained_snn './trained_models/snn/snn_resnet20_cifar100_5_2020-11-03 15:40:53.778615.pth' --log &

#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 150 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 0.3 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.2 --train_acc_batches 500 --devices 0 --pretrained_ann './trained_models/ann/ann_vgg16_cifar10_sgd100.2.pth' --log & 

#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 50 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 0.4 --optimizer Adam --weight_decay 0 --amsgrad True --betas '0,0.999' --dropout 0 --train_acc_batches 500 --devices 0 --pretrained_ann './trained_models/ann/ann_vgg16_cifar10_sgd100.2.pth' --log &

#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG5 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 10 --timesteps 20 --leak 1.0 --scaling_factor 0.7 --default_threshold 0.8 --optimizer Adam --weight_decay 0 --momentum 0.9 --amsgrad True --dropout 0 --train_acc_batches 50 --devices 2 --pretrained_ann './trained_models/ann/ann_vgg5_cifar10_pbr.pth' #--test_only --test_acc_every_batch #--log &

#python snn.py --dataset IMAGENET --batch_size 16 --architecture VGG16 --learning_rate 5e-7 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 5 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0 --train_acc_batches 5000 --devices 0,1,2,3 --pretrained_snn './trained_models/snn/snn_vgg16_imagenet_5_2020-11-15 17:19:05.906871_train.pth' --log & #--test_only --test_acc_every_batch #--log &

#python snn.py --dataset IMAGENET --batch_size 12 --architecture VGG16 --learning_rate 1e-4 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 5 --leak 1.0 --scaling_factor 0.9 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0 --train_acc_batches 50 --devices 0,1,2,3 --pretrained_snn '/home/nano01/a/rathi2/trained_models/SNN/snn_vgg16_imagenet_25_2020-07-18 18:29:29.363776_train.pth' --dont_save #--log & #--test_only --test_acc_every_batch #--log &

#python snn_imagenet.py --dataset IMAGENET --batch_size 20 --architecture VGG16 --learning_rate 1e-4 --epochs 2 --lr_interval '0.60 0.80 0.90' --lr_reduce 4 --timesteps 25 --leak 1.0 --scaling_factor 0.7 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 0,1,2,3 --default_threshold 0.5 --pretrained_ann './trained_models/ann/best_ann_vgg16_imagenet.pth' --test_only --test_acc_every_batch --dont_save #

#sleep 12330
#python snn_imagenet.py --dataset IMAGENET --batch_size 4 --architecture VGG16 --learning_rate 1e-7 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 25 --leak 1.0 --scaling_factor 0.7 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 0,1,2,3 --default_threshold 0.5 --resume './trained_models/snn/snn_vgg16_imagenet_25_2020-07-13 12:05:06.041433_train.pth' --log & #--test_only --test_acc_every_batch #

#python snn_imagenet.py --dataset IMAGENET --batch_size 40 --architecture VGG16 --learning_rate 1e-4 --epochs 2 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 25 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 0,1,2,3 --default_threshold 0.5 --pretrained_snn './trained_models/snn/snn_vgg16_imagenet_25_2020-07-18 18:29:29.363776_train.pth' --test_only --test_acc_every_batch --dont_save

#python snn.py --dataset CIFAR100 --batch_size 64 --architecture RESNET34 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 25 --leak 1.0 --scaling_factor 0.6 --default_threshold 0.9 --optimizer Adam --weight_decay 0 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 500 --devices 0 --pretrained_ann './trained_models/ann/ann_resnet34_cifar100.pth' --log & #--test_only --test_acc_every_batch

#python snn.py --dataset CIFAR100 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 0.6 --optimizer Adam --weight_decay 0 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 500 --devices 1 --pretrained_ann './trained_models/ann/ann_vgg16_cifar100.pth' --log & #--test_only --test_acc_every_batch

#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG5 --learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 1.0 --default_threshold 0.9 --optimizer Adam --weight_decay 1e-4 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 500 --devices 2 --log &

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#ResNet ImageNet

#python snn_imagenet.py --dataset IMAGENET --batch_size 15 --architecture RESNET34 --learning_rate 1e-4 --epochs 2 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 45 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 0 --default_threshold 0.5 --pretrained_snn './trained_models/snn/snn_resnet34_imagenet_35_2020-07-22 01:40:48.547098_train.pth' --test_only --test_acc_every_batch --dont_save

#python snn_imagenet.py --dataset IMAGENET --batch_size 3 --architecture RESNET34 --learning_rate 1e-4 --epochs 5 --lr_interval '0.60 0.80 0.90' --lr_reduce 4 --timesteps 35 --leak 1.0 --scaling_factor 0.7 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 0,1,2 --default_threshold 0.95 --pretrained_ann './trained_models/ann/best_ann_resnet34_imagenet.pth' --log & #--test_only --test_acc_every_batch #

#python snn_imagenet.py --dataset IMAGENET --batch_size 4 --architecture RESNET34 --learning_rate 1e-4 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 2 --timesteps 35 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0 --momentum 0 --amsgrad True --dropout 0.1 --train_acc_batches 1024 --devices 1,2,3,0 --default_threshold 0.5 --resume './trained_models/snn/snn_resnet34_imagenet_35_2020-07-04 00:04:37.838974_train.pth' --log & #--test_only --test_acc_every_batch #

#__________________________________________________________________________________________________________________________________________________________________

#python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 --learning_rate 1e-4 --epochs 100 --lr_interval '0.50 0.70 0.90' --lr_reduce 5 --timesteps 20 --leak 1.0 --scaling_factor 0.5 --optimizer Adam --weight_decay 0 --amsgrad True --dropout 0.1 --train_acc_batches 500 --devices 2 --pretrained_ann './trained_models/ann/ann_vgg16_cifar10.pth' --log & #--test_only --test_acc_every_batch --dont_save
