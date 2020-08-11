#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from self_models import *
import sys
import os
import shutil
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):
    
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    model.module.network_update(timesteps=timesteps, leak=1.0)
    pos=0
    thresholds=[]
    #pre_calculated = [7.612769603729248, 12.23182201385498, 2.7110540866851807, 2.30560040473938, 0.7672547101974487, 1.658124566078186, 1.81326162815094, 0.5426592826843262, 0.9945646524429321, 1.4132213592529297, 0.3058353066444397, 1.045074224472046, 1.0654196739196777, 0.13718076050281525, 0.6044105291366577]

    def find(layer):
        max_act=0
        
        f.write('\n Finding threshold for layer {}'.format(layer))
        # if layer <=43:
        #     thresholds.append(pre_calculated.pop(0))
        #     f.write(' {}'.format(thresholds))
        #     model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
        #     return

        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output.max()>max_act:
                    max_act = output.max().item()

                    #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==1:
                    thresholds.append(max_act)
                    f.write(' {}'.format(thresholds))
                    model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
    
    if architecture.lower().startswith('vgg'):              
        for l in model.module.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(c[0]) == len(model.module.classifier) -1):
                    break
                else:
                    find(int(l[0])+int(c[0])+1)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        pos = len(model.module.pre_process)

        for i in range(1,5):
            layer = model.module.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].delay_path)):
                    if isinstance(layer[index].delay_path[l],nn.Conv2d):
                        pos = pos +1

        for c in model.module.classifier.named_children():
            if isinstance(c[1],nn.Linear):
                if (int(c[0])==len(model.module.classifier)-1):
                    break
                else:
                    find(int(c[0])+pos)

    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    global learning_rate
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch>start_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))
    
    #total_loss = 0.0
    #total_correct = 0
    model.train()
    local_time = datetime.datetime.now()   
    #time_chunks = timesteps
    #model.module.network_update(timesteps=time_chunks, leak=leak)
    #current_time = start_time
    #model.module.network_init(update_interval)
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if epoch == start_epoch and start_batch > 300000:
            break
        #if batch_idx < start_batch:
        #    continue
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        if (batch_idx+1)%32==0:
            #pdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()
        
        t = 0
        mem = 0
        mask = 0
        spike = 0
        #pdb.set_trace()
        # while t<(timesteps-time_chunks):
        #     with torch.no_grad():
        #         output, mem, spike, mask = model(data, mem=mem, spike=spike, mask=mask)
        #         t = t+time_chunks
        
        output, mem, spike, mask = model(data, mem=mem, spike=spike, mask=mask)
        #pdb.set_trace()
        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()
        loss.backward()
               
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        
        for key, value in model.module.leak.items():
            # maximum of leak=1.0
            model.module.leak[key].data.clamp_(max=1.0)

        losses.update(loss.item(),data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
                
        if (batch_idx+1) % train_acc_batches == 0:
            temp1 = []
            temp2 = []
            
            for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[round(value.item(),2)]
            for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[round(value.item(),2)]
            f.write('\n\n Epoch: {}, lr: {:.1e}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}, time: {}'
                    .format(epoch,
                        learning_rate,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        temp2,
                        timesteps,
                        datetime.timedelta(seconds=(datetime.datetime.now() - local_time).seconds)
                        )
                    )
            local_time = datetime.datetime.now()
        
        if (batch_idx+1) % 12500 == 0:    
            temp1 = []
            temp2 = []
            for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[value.item()]
            for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[value.item()]
            state = {
                    'epoch'                 : epoch,
                    'batch'                 : batch_idx,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : temp2,
                    'activation'            : activation,
                    'accuracy'              : top1.avg
                }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass 
            filename = './trained_models/snn/'+identifier+'_train.pth'
            torch.save(state,filename)
        
        # if (batch_idx+1)%25000 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] / lr_reduce
        #         learning_rate = param_group['lr']

    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        )
                    )
      
def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    
    temp1 = []
    temp2 = []
    for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
        temp1 = temp1+[round(value.item(),2)]
    for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
        temp2 = temp2+[round(value.item(),2)]
    print('\n Thresholds: {}, leak: {}'.format(temp1, temp2))

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            mem = 0
            mask = 0
            spike = 0
            output,_,_,_    = model(data, mem=mem, spike=spike, mask=mask) 
            loss            = F.cross_entropy(output,target)
            pred            = output.max(1,keepdim=True)[1]
            correct         = pred.eq(target.data.view_as(pred)).cpu().sum()
            #print(output.min())
            #pdb.set_trace()
            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                f.write('\n Images {}/{} Loss: {:.4f} Accuracy: {}/{}({:.4f})'
                    .format(
                    test_loader.batch_size*(batch_idx+1),
                    len(test_loader.dataset),
                    losses.avg,
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
            #if batch_idx==100:
            #    break
        
        temp1 = []
        temp2 = []
        for key, value in sorted(model.module.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[value.item()]
        for key, value in sorted(model.module.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[value.item()]
        
        if epoch>5 and top1.avg<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
             
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : temp2,
                    'activation'            : activation
                }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass 
            filename = './trained_models/snn/'+identifier+'_test.pth'
            if not args.dont_save:
                torch.save(state,filename)    
        
            #if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=30,                 type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=20,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.3,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.95,               type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.5,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=1000,               type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--resume',                 default='',                 type=str,       help='resume training from this state')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches
    resume              = args.resume
    start_epoch         = 1
    start_batch         = 0
    max_accuracy        = 0.0

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/snn/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+str(datetime.datetime.now())
    log_file+=identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file

    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    elif dataset == 'IMAGENET':
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
    
    elif dataset == 'MNIST':
        trainset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        testset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

    elif dataset == 'IMAGENET':
        labels      = 1000
        traindir    = os.path.join('/local/a/imagenet/imagenet2012/', 'train')
        valdir      = os.path.join('/local/a/imagenet/imagenet2012/', 'val')
        trainset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])) 

    train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=True)

    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN_STDB_IMAGENET(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    
    elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN_STDB_IMAGENET(resnet_name = architecture, activation = activation, labels=labels, timesteps=timesteps,leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, dataset=dataset)

    # if freeze_conv:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model)   
    #model = nn.parallel.DistributedDataParallel(model)
    #pdb.set_trace()

    if pretrained_ann:
      
        state = torch.load(pretrained_ann, map_location='cpu')
        
        if architecture.lower().startswith('res'):
            missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
            #pdb.set_trace()
        else:
            cur_dict = model.state_dict()
            cur_dict['module.features.0.weight']    = nn.Parameter(state['state_dict']['features.module.0.weight'].data)
            cur_dict['module.features.3.weight']    = nn.Parameter(state['state_dict']['features.module.3.weight'].data)
            cur_dict['module.features.6.weight']    = nn.Parameter(state['state_dict']['features.module.6.weight'].data)
            cur_dict['module.features.9.weight']    = nn.Parameter(state['state_dict']['features.module.9.weight'].data)
            cur_dict['module.features.12.weight']   = nn.Parameter(state['state_dict']['features.module.12.weight'].data)
            cur_dict['module.features.15.weight']   = nn.Parameter(state['state_dict']['features.module.15.weight'].data)
            cur_dict['module.features.18.weight']   = nn.Parameter(state['state_dict']['features.module.18.weight'].data)
            cur_dict['module.features.21.weight']   = nn.Parameter(state['state_dict']['features.module.21.weight'].data)
            cur_dict['module.features.24.weight']   = nn.Parameter(state['state_dict']['features.module.24.weight'].data)
            cur_dict['module.features.27.weight']   = nn.Parameter(state['state_dict']['features.module.27.weight'].data)
            cur_dict['module.features.30.weight']   = nn.Parameter(state['state_dict']['features.module.30.weight'].data)
            cur_dict['module.features.33.weight']   = nn.Parameter(state['state_dict']['features.module.33.weight'].data)
            cur_dict['module.features.36.weight']   = nn.Parameter(state['state_dict']['features.module.36.weight'].data)
            cur_dict['module.classifier.0.weight']  = nn.Parameter(state['state_dict']['classifier.0.weight'].data)
            cur_dict['module.classifier.3.weight']  = nn.Parameter(state['state_dict']['classifier.3.weight'].data)
            cur_dict['module.classifier.6.weight']  = nn.Parameter(state['state_dict']['classifier.6.weight'].data)
            missing_keys, unexpected_keys = model.load_state_dict(cur_dict, strict=False)
        
        f.write('\n Missing keys: {}'.format(missing_keys))
        f.write('\n Unexpected keys: {}'.format(unexpected_keys))
        f.write('\n Info: Top-1 (Top-5) accuracy of loaded ANN model: {}({})'.format(state['acc1'], state['acc5']))

        #Freezing all weight layers and only training for leak and threshold
        # for param in model.parameters():
        #     if param.shape[0]>1:
        #         param.requires_grad = False
        

        #If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            f.write('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        # Find the threhsolds and save it to the ANN file for further use
        else:
            thresholds = find_threshold(batch_size=32, timesteps=500, architecture=architecture)
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            
            #Save the threhsolds in the ANN file
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)
    
    elif pretrained_snn:
                
        state = torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)

        # cur_dict = model.state_dict()
        # cur_dict['module.features.0.weight']    = nn.Parameter(state['state_dict']['module.features.0.weight'].data)
        # cur_dict['module.features.3.weight']    = nn.Parameter(state['state_dict']['module.features.3.weight'].data)
        # cur_dict['module.features.6.weight']    = nn.Parameter(state['state_dict']['module.features.6.weight'].data)
        # cur_dict['module.features.9.weight']    = nn.Parameter(state['state_dict']['module.features.9.weight'].data)
        # cur_dict['module.features.12.weight']   = nn.Parameter(state['state_dict']['module.features.12.weight'].data)
        # cur_dict['module.features.15.weight']   = nn.Parameter(state['state_dict']['module.features.15.weight'].data)
        # cur_dict['module.features.18.weight']   = nn.Parameter(state['state_dict']['module.features.18.weight'].data)
        # cur_dict['module.features.21.weight']   = nn.Parameter(state['state_dict']['module.features.21.weight'].data)
        # cur_dict['module.features.24.weight']   = nn.Parameter(state['state_dict']['module.features.24.weight'].data)
        # cur_dict['module.features.27.weight']   = nn.Parameter(state['state_dict']['module.features.27.weight'].data)
        # cur_dict['module.features.30.weight']   = nn.Parameter(state['state_dict']['module.features.30.weight'].data)
        # cur_dict['module.features.33.weight']   = nn.Parameter(state['state_dict']['module.features.33.weight'].data)
        # cur_dict['module.features.36.weight']   = nn.Parameter(state['state_dict']['module.features.36.weight'].data)
        # cur_dict['module.classifier.0.weight']  = nn.Parameter(state['state_dict']['module.classifier.0.weight'].data)
        # cur_dict['module.classifier.3.weight']  = nn.Parameter(state['state_dict']['module.classifier.3.weight'].data)
        # cur_dict['module.classifier.6.weight']  = nn.Parameter(state['state_dict']['module.classifier.6.weight'].data)

        # missing_keys, unexpected_keys = model.load_state_dict(cur_dict, strict=False)
        f.write('\n Missing keys: {}, Unexpected keys: {}'.format(missing_keys, unexpected_keys))
        # thresholds = state['thresholds']
        # leak       = state['leak']
        # model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
        # model.module.network_update(timesteps=timesteps, leak=leak)
    
    f.write('\n {}'.format(model))
        
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)

    # find_threshold() alters the timesteps and leak, restoring it here
    model.module.network_update(timesteps=timesteps, leak=leak)
        
    f.write('\n {}'.format(optimizer))

    if resume:
        f.write('\n Resuming from checkpoint {}'.format(resume))
        state = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        
        # cur_dict = model.state_dict()
        # cur_dict['module.features.0.weight']    = nn.Parameter(state['state_dict']['module.features.0.weight'].data)
        # cur_dict['module.features.3.weight']    = nn.Parameter(state['state_dict']['module.features.3.weight'].data)
        # cur_dict['module.features.6.weight']    = nn.Parameter(state['state_dict']['module.features.6.weight'].data)
        # cur_dict['module.features.9.weight']    = nn.Parameter(state['state_dict']['module.features.9.weight'].data)
        # cur_dict['module.features.12.weight']   = nn.Parameter(state['state_dict']['module.features.12.weight'].data)
        # cur_dict['module.features.15.weight']   = nn.Parameter(state['state_dict']['module.features.15.weight'].data)
        # cur_dict['module.features.18.weight']   = nn.Parameter(state['state_dict']['module.features.18.weight'].data)
        # cur_dict['module.features.21.weight']   = nn.Parameter(state['state_dict']['module.features.21.weight'].data)
        # cur_dict['module.features.24.weight']   = nn.Parameter(state['state_dict']['module.features.24.weight'].data)
        # cur_dict['module.features.27.weight']   = nn.Parameter(state['state_dict']['module.features.27.weight'].data)
        # cur_dict['module.features.30.weight']   = nn.Parameter(state['state_dict']['module.features.30.weight'].data)
        # cur_dict['module.features.33.weight']   = nn.Parameter(state['state_dict']['module.features.33.weight'].data)
        # cur_dict['module.features.36.weight']   = nn.Parameter(state['state_dict']['module.features.36.weight'].data)
        # cur_dict['module.classifier.0.weight']  = nn.Parameter(state['state_dict']['module.classifier.0.weight'].data)
        # cur_dict['module.classifier.3.weight']  = nn.Parameter(state['state_dict']['module.classifier.3.weight'].data)
        # cur_dict['module.classifier.6.weight']  = nn.Parameter(state['state_dict']['module.classifier.6.weight'].data)
        
        # missing_keys, unexpected_keys = model.load_state_dict(cur_dict, strict=False)
        f.write('\n Missing keys: {}, Unexpected keys: {}'.format(missing_keys, unexpected_keys))
        # thresholds = state['thresholds']
        # leak       = state['leak']
        # model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
        # model.module.network_update(timesteps=timesteps, leak=leak)
        
        # if optimizer == 'Adam':
        #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
        # elif optimizer == 'SGD':
        #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)
        
        start_epoch     = state['epoch']
        batch           = state['batch']
        start_batch     = batch+1
        accuracy        = state['accuracy']
        #pdb.set_trace()
        optimizer.load_state_dict(state['optimizer'])
        for param_group in optimizer.param_groups:
            learning_rate =  param_group['lr']
            #param_group['lr'] = learning_rate
        #start_epoch = start_epoch + 1
        #start_batch = 0
        f.write('\n Loaded from resume epoch: {}, batch: {} accuracy: {:.4f} lr: {:.1e}'.format(start_epoch, batch, accuracy, learning_rate))
    
    # else:
    #     f.write('\n {}'.format(model))
        
    #     if torch.cuda.is_available() and args.gpu:
    #         model.cuda()
    
    #     if optimizer == 'Adam':
    #         optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    #     elif optimizer == 'SGD':
    #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)
        
    #     # find_threshold() alters the timesteps and leak, restoring it here
    #     model.module.network_update(timesteps=timesteps, leak=leak)
        
    #     f.write('\n {}'.format(optimizer))

    for epoch in range(start_epoch, epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))




