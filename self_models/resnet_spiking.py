import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
import copy

cfg = {
	'resnet6'	: [1,1,0,0],
	'resnet12' 	: [1,1,1,1],
	'resnet20'	: [2,2,2,2],
	'resnet34'	: [3,4,6,3]
}

class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike):
        
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(input).cuda()
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad*grad_input, None

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout):
        #print('In __init__ BasicBlock')
        #super(BasicBlock, self).__init__()
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, dic):
        #print('In forward BasicBlock')
        #pdb.set_trace()
        out_prev 		= dic['out_prev']
        pos 			= dic['pos']
        act_func 		= dic['act_func']
        mem 			= dic['mem']
        spike 			= dic['spike']
        mask 			= dic['mask']
        threshold 		= dic['threshold']
        t 				= dic['t']
        leak			= dic['leak']
        #find_max_mem 	= dic['find_max_mem']
        inp				= out_prev.clone()
        # for m in mem:
        # 	m.detach_()
        # for s in spike:
        # 	s.detach_()

        #conv1
        delta_mem		= self.residual[0](inp)
        mem[pos] 		= getattr(leak, 'l'+str(pos)) *mem[pos] + delta_mem
        mem_thr 		= (mem[pos]/getattr(threshold, 't'+str(pos))) - 1.0
        rst 			= getattr(threshold, 't'+str(pos)) * (mem_thr>0).float()
        #mem[pos] 		= getattr(leak, 'l'+str(pos)) *mem[pos] + self.residual[0](inp) - rst
        mem[pos] 		= mem[pos] - rst

        #relu1
        out 			= act_func(mem_thr, (t-1-spike[pos]))
        spike[pos] 		= spike[pos].masked_fill(out.bool(),t-1)
        out_prev  		= out.clone()
	
		#dropout1
        out_prev 		= out_prev * mask[pos]
		
		#conv2+identity
        delta_mem 		= self.residual[3](out_prev) + self.identity(inp)
        mem[pos+1] 		= getattr(leak, 'l'+str(pos+1))*mem[pos+1] + delta_mem
        mem_thr 		= (mem[pos+1]/getattr(threshold, 't'+str(pos+1))) - 1.0
        rst 			= getattr(threshold, 't'+str(pos+1)) * (mem_thr>0).float()
        #mem[pos+1] 		= getattr(leak, 'l'+str(pos+1))*mem[pos+1] + self.residual[3](out_prev) + self.identity(inp) - rst
        mem[pos+1] 		= mem[pos+1] - rst

        #relu2
        out 			= act_func(mem_thr, (t-1-spike[pos+1]))
        spike[pos+1]	= spike[pos+1].masked_fill(out.bool(),t-1)
        out_prev  		= out.clone()

        #if find_max_mem:
        #	return (self.delay_path[2](out_prev) + self.shortcut(inp)).max()
        #if t==199:
        #	print((self.delay_path[3](out_prev) + self.shortcut(inp)).max())
        #if len(self.shortcut)>0:
        
        
        #else:
        #	mem[1] 		= leak_mem*mem[1] + self.delay_path[1](out_prev) + inp - rst
        
        
        
        #result				= {}
        #result['out_prev'] 	= out.clone()
        #result['mem'] 		= mem[:]
        #result['spike'] 	= spike[:]
        
        #pdb.set_trace()
        return out_prev

class RESNET_SNN_STDB(nn.Module):
	
	#all_layers = []
	#drop 		= 0.2
	def __init__(self, resnet_name, activation='Linear', labels=10, timesteps=75, leak=1.0, default_threshold=1.0, alpha=0.5, beta=0.035, dropout=0.2, dataset='CIFAR10'):

		super().__init__()
		
		self.resnet_name	= resnet_name.lower()
		if activation == 'Linear':
			self.act_func 	= LinearSpike.apply
		elif activation == 'STDB':
			self.act_func	= STDB.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		STDB.alpha 			= alpha
		STDB.beta 			= beta 
		self.dropout 		= dropout
		self.dataset 		= dataset
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		

		self.pre_process    = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.AvgPool2d(2)
                                )
		block 				= BasicBlock
		self.in_planes      = 64
		
		self.layer1 		= self._make_layer(block, 64, cfg[self.resnet_name][0], stride=1, dropout=self.dropout)
		self.layer2 		= self._make_layer(block, 128, cfg[self.resnet_name][1], stride=2, dropout=self.dropout)
		self.layer3 		= self._make_layer(block, 256, cfg[self.resnet_name][2], stride=2, dropout=self.dropout)
		self.layer4 		= self._make_layer(block, 512, cfg[self.resnet_name][3], stride=2, dropout=self.dropout)
		#self.avgpool 		= nn.AvgPool2d(2)
		
		self.classifier     = nn.Sequential(
									nn.Linear(512*2*2, labels, bias=False)
									)
		
		# self.classifier     = nn.Sequential(
  #                               nn.Linear(512*2*2, 1024, bias=False),
  #                               nn.ReLU(inplace=True),
  #                               nn.Dropout(self.dropout),
  #                               nn.Linear(1024, 1024, bias=False),
  #                               nn.ReLU(inplace=True),
  #                               nn.Dropout(self.dropout),
  #                               nn.Linear(1024, labels, bias=False)
  #                               )

		self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4:self.layer4}

		self._initialize_weights2()
		
		threshold 	= {}
		lk 			= {}
		for l in range(len(self.pre_process)):
			if isinstance(self.pre_process[l],nn.Conv2d):
				#self.register_buffer('threshold[l]', torch.tensor(default_threshold, requires_grad=True))
				threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(l)] 	  		= nn.Parameter(torch.tensor(leak))

		pos = len(self.pre_process)
				
		for i in range(1,5):

			layer = self.layers[i]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						threshold['t'+str(pos)] = nn.Parameter(torch.tensor(default_threshold))
						lk['l'+str(pos)] 		= nn.Parameter(torch.tensor(leak))
						pos=pos+1

		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				threshold['t'+str(pos+l)] 		= nn.Parameter(torch.tensor(default_threshold))
				lk['l'+str(pos+l)] 				= nn.Parameter(torch.tensor(leak)) 
				
		self.threshold 	= nn.ParameterDict(threshold)
		self.leak 		= nn.ParameterDict(lk)
		
	def _initialize_weights2(self):

		for m in self.modules():
			
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):
    	
		self.scaling_factor = scaling_factor

		# for key in sorted(self.threshold.keys()):
		# 	if thresholds:
		# 		self.threshold[key].data = torch.tensor(thresholds.pop(0)*self.scaling_factor)
		
		for pos in range(len(self.pre_process)):
			if isinstance(self.pre_process[pos],nn.Conv2d):
				if thresholds:
					self.threshold.update({'t'+str(pos): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})

		pos = len(self.pre_process)
		for i in range(1,5):
			layer = self.layers[i]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						#self.threshold[pos].data = torch.tensor(thresholds.pop(0)*self.scaling_factor)
						pos = pos+1

		for l in range(len(self.classifier)):
			if isinstance(self.classifier[l], nn.Linear):
				if thresholds:
					self.threshold.update({'t'+str(pos+l): nn.Parameter(torch.tensor(thresholds.pop(0)*self.scaling_factor))})

		
		# pos = len(self.pre_process)
		# while(True):
		# 	try:
		# 		self.threshold[pos].data = torch.tensor(self.threshold[pos].data * scaling_factor)
		# 		pos=pos+1
		# 	except:
		# 		break

	def _make_layer(self, block, planes, num_blocks, stride, dropout):

		if num_blocks==0:
			return nn.Sequential()
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, dropout))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
		# for key, value in sorted(self.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
		# 	if isinstance(leak, list) and leak:
		# 		self.leak.update({key: nn.Parameter(torch.tensor(leak.pop(0)))})
	
	def neuron_init(self, x):
		
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)

		self.mem 	= {}
		self.spike 	= {}
		self.mask 	= {}

		# Pre process layers
		for l in range(len(self.pre_process)):
			
			if isinstance(self.pre_process[l], nn.Conv2d):
				self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height)
				self.spike[l] = torch.ones(self.mem[l].shape)*(-1000)
				#self.register_buffer('mem[l]', torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height))
				#self.register_buffer('spike[l]', torch.ones(self.mem[l].shape)*(-1000))

			elif isinstance(self.pre_process[l], nn.Dropout):
				self.mask[l] = self.pre_process[l](torch.ones(self.mem[l-2].shape))
			elif isinstance(self.pre_process[l], nn.AvgPool2d):
				
				self.width 	= self.width//self.pre_process[l].kernel_size
				self.height = self.height//self.pre_process[l].kernel_size 

		pos = len(self.pre_process)
		for i in range(1,5):
			layer = self.layers[i]
			self.width = self.width//layer[0].residual[0].stride[0]
			self.height = self.height//layer[0].residual[0].stride[0]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
						self.spike[pos] = torch.ones(self.mem[pos].shape)*(-1000)
						#self.register_buffer('mem[pos]', torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height))
						#self.register_buffer('spike[pos]', torch.ones(self.mem[pos].shape)*(-1000))
						pos = pos + 1
					elif isinstance(layer[index].residual[l],nn.Dropout):
						self.mask[pos-1] = layer[index].residual[l](torch.ones(self.mem[pos-1].shape))
		
		#average pooling before final layer
		#self.width 	= self.width//self.avgpool.kernel_size
		#self.height = self.height//self.avgpool.kernel_size

		#final classifier layer
		#self.mem[pos] = torch.zeros(self.batch_size, self.classifier[0].out_features)
		#self.register_buffer('mem[pos]', torch.zeros(self.batch_size, self.classifier[0].out_features))

		# self.spike = copy.deepcopy(self.mem)
		# for key in self.spike.keys():
		# 	self.spike[key].fill_(-1000)

		for l in range(len(self.classifier)):
			if isinstance(self.classifier[l],nn.Linear):
				self.mem[pos+l] 	= torch.zeros(self.batch_size, self.classifier[l].out_features)
				self.spike[pos+l] 	= torch.ones(self.mem[pos+l].shape)*(-1000)
			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[pos+l] 	= self.classifier[l](torch.ones(self.mem[pos+l-2].shape))

	def percentile(self, t, q):
		k = 1 + round(.01 * float(q) * (t.numel() - 1))
		result = t.view(-1).kthvalue(k).values.item()
		return result

	def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
		self.neuron_init(x)
			
		max_mem = 0.0
		#pdb.set_trace()
		for t in range(self.timesteps):

			out_prev = x
					
			for l in range(len(self.pre_process)):
							
				if isinstance(self.pre_process[l], nn.Conv2d):
					
					if find_max_mem and l==max_mem_layer:
						cur = self.percentile(self.pre_process[l](out_prev).view(-1), 99.7)
						if (cur>max_mem):
							max_mem = torch.tensor([cur])
						break
					
					delta_mem 		= self.pre_process[l](out_prev)
					self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + delta_mem
					mem_thr 		= (self.mem[l]/getattr(self.threshold, 't'+str(l))) - 1.0
					rst 			= getattr(self.threshold, 't'+str(l)) * (mem_thr>0).float()
					#self.mem[l] 	= getattr(self.leak, 'l'+str(l)) *self.mem[l] + self.pre_process[l](out_prev) - rst
					self.mem[l] 	= self.mem[l] - rst
					
				elif isinstance(self.pre_process[l], nn.ReLU):
					out 			= self.act_func(mem_thr, (t-1-self.spike[l-1]))
					self.spike[l-1] 	= self.spike[l-1].masked_fill(out.bool(),t-1)
					out_prev  		= out.clone()

				elif isinstance(self.pre_process[l], nn.AvgPool2d):
					out_prev 		= self.pre_process[l](out_prev)
				
				elif isinstance(self.pre_process[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]
			
			if find_max_mem and max_mem_layer<len(self.pre_process):
				continue
				
			pos 	= len(self.pre_process)
			
			for i in range(1,5):
				layer = self.layers[i]
				for index in range(len(layer)):
					out_prev = layer[index]({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem, 'spike':self.spike, 'mask':self.mask, 'threshold':self.threshold, 't': t, 'leak':self.leak})
					pos = pos+2
			
			#out_prev = self.avgpool(out_prev)
			out_prev = out_prev.reshape(self.batch_size, -1)

			for l in range(len(self.classifier)-1):
				
				if isinstance(self.classifier[l], (nn.Linear)):
					if find_max_mem and (pos+l)==max_mem_layer:
						if (self.classifier[l](out_prev)).max()>max_mem:
							max_mem = (self.classifier[l](out_prev)).max()
						break

					mem_thr 			= (self.mem[pos+l]/getattr(self.threshold, 't'+str(pos+l))) - 1.0
					out 				= self.act_func(mem_thr, (t-1-self.spike[pos+l]))
					rst 				= getattr(self.threshold, 't'+str(pos+l)) * (mem_thr>0).float()
					self.spike[pos+l] 	= self.spike[pos+l].masked_fill(out.bool(),t-1)
					self.mem[pos+l] 	= getattr(self.leak, 'l'+str(pos+l))*self.mem[pos+l] + self.classifier[l](out_prev) - rst
					out_prev  			= out.clone()

				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[pos+l]

			#pdb.set_trace()
			# Compute the final layer outputs
			if not find_max_mem:
				if len(self.classifier)>1:
					self.mem[pos+l+1] 		= self.mem[pos+l+1] + self.classifier[l+1](out_prev)
				else:
					self.mem[pos] 			= self.mem[pos] + self.classifier[0](out_prev)
		
		if find_max_mem:
			return max_mem

		if len(self.classifier)>1:
			return self.mem[pos+l+1]
		else:
			return self.mem[pos]	
