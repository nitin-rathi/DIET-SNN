#####################################
#   @author: Nitin Rathi    		#
#	Writes commands to execute		#
# 	in script.sh				 	#
#####################################

import os
import itertools
import pdb
import argparse
from scipy.special import comb

#For SNN set architecture sequentially because it has to mapped to the trained ANN file
#For SNN timesteps has a direct relationship wiht scaling_factor
#So for samll timesteps the scaling_factor should be small
#For timesteps above 100 scaling_factor of 0.6-0.8 can be used

pretrained_ann = './trained_models/ann/ann_resnet20_cifar100.pth'
hyperparameters = {
	'dataset'		:	{'CIFAR100'},
	'batch_size'	:	{'64'},
	'architecture'	:	{'RESNET20'},
	'learning_rate'	:	{'1e-2'},
	'epochs'		:	{'30'},
	'lr_interval'	:	{'\'0.60 0.80 0.90\''},
	'lr_reduce'		: 	{'10'},
	'timesteps'		: 	{'20'},
	'leak' 			: 	{'1.0'},
	'scaling_factor':	{'0.3'},
	'optimizer' 	: 	{'SGD'},
	'weight_decay'	:	{'5e-4'},
	'momentum' 		: 	{'0.95'},
	'dropout' 		: 	{'0.3'},
	'train_acc_batches'	: 	{'1000'},
	'devices' 		:	{'1'}

}

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='create script for hyperparameter optimization')
	parser.add_argument('--filename', 		default='snn.py',		help='python filename to run')
	parser.add_argument('--parallel',		action='store_true',	help='whether to allow all combinations to run simultaneously')
	args = parser.parse_args()

	f = open('script.sh', 'w', buffering=1)
	f.write('#!/bin/bash')
	f.write('\n')
	
	keys, values = zip(*hyperparameters.items())
	combinations = [dict(zip(keys,v)) for v in itertools.product(*values)]
	print('Total possible combinations: ',len(combinations))
	
	for c in combinations:
		s = ''
		for key, value in c.items():
			s = s+'--'+key+' '+value+' '
			
		s = 'python '+args.filename+' '+s
		s = s+'--pretrained_ann '+'\''+pretrained_ann+'\' '
		s = s+'--log '
		if args.parallel:
			s = s + '& '
		f.write('\n')
		f.write(s)
	
	f.close()
	#os.system('./script.sh')