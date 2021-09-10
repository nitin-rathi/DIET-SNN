# Files:
ann.py - Trains an ANN

snn.py - Trains an SNN

# To train ANN:
python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG9 --learning\_rate 1e-3 --epochs 300 --optimizer Adam --dropout 0.2 --devices 2 --log

# Convert ANN to SNN and train SNN
python snn.py --dataset CIFAR10 --batch_size 64 --architecture VGG9 --learning\_rate 1e-4 --optimizer 'Adam' --epochs 300 --timesteps 5 --scaling\_factor 0.2 --weight\_decay 0 --dropout 0.2 --train\_acc\_batches 500 --devices 2 --default\_threshold 0.4 --pretrained\_ann './trained\_models/ann/ann\_vgg9\_cifar10.pth' --log
