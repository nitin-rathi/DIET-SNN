Files:
	ann.py - Trains an ANN
	snn.py - Trains an SNN

To train ANN:
	python ann.py --dataset CIFAR100 --architecture RESNET20 --batch_size 64 --learning_rate 1e-2 --epochs 225 --lr_interval '0.50 0.70 0.90' --lr_reduce 5 --optimizer SGD --dropout 0.3

Convert ANN to SNN and train SNN
	python snn.py --dataset CIFAR100 --batch_size 64 --architecture RESNET20 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 10 --timesteps 25 --leak 1.0 --scaling_factor 0.7 --default_threshold 0.8 --optimizer Adam --weight_decay 0 --amsgrad True --dropout 0.1 --train_acc_batches 1 --pretrained_ann './trained_models/ann/ann_resnet20_cifar100.pth'

Use the pre-trained ANN model and convert ANN to SNN and train SNN
	python snn.py --dataset CIFAR100 --batch_size 64 --architecture RESNET20 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 10 --timesteps 25 --leak 1.0 --scaling_factor 0.7 --default_threshold 0.8 --optimizer Adam --weight_decay 0 --amsgrad True --dropout 0.1 --train_acc_batches 1 --pretrained_ann './trained_models/ann_resnet20_cifar100.pth'

Use the pre-trained ANN model and convert ANN to SNN and test SNN without optimizing threshold and leak
	python snn.py --dataset CIFAR100 --batch_size 64 --architecture RESNET20 --learning_rate 1e-4 --epochs 100 --lr_interval '0.60 0.80 0.90' --lr_reduce 10 --timesteps 25 --leak 1.0 --scaling_factor 0.7 --default_threshold 0.8 --optimizer Adam --weight_decay 0 --amsgrad True --dropout 0.1 --train_acc_batches 1 --pretrained_ann './trained_models/ann_resnet20_cifar100.pth' --test_only --test_acc_every_batch