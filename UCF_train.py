import os
import sys
import argparse
from datetime import datetime
import time

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

#torchvision
from torchvision import transforms
from data.UCF101_dataset.dataloader import get_dataloader

import numpy as np
from i3d import I3D,Unit3Dpy
from utils.print_time import print_time
from utils.util import AverageMeter, calculate_accuracy

def train(args):
	#prepare dataset
	dataloaders = get_dataloader(args)
	print('------------data loaded -------------')
	
	#model
	load_model_start = time.localtime(time.time())
	i3d = I3D(num_classes = 400)
	if args.mode == 'rgb':
		i3d.load_state_dict(torch.load('model/model_rgb.pth'))
	elif args.mode == 'flow':
		i3d.load_state_dict(torch.load('model/model_flow.pth'))
	else:
		raise ValueError('mode excepted to be [rgb|flow],but get{}'.format(mode))
	i3d.conv3d_0c_1x1 = Unit3Dpy(1024, args.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), activation=None, use_bias=True, use_bn=False)
	if bool(args.use_cuda):
		i3d.cuda()
	print('------------model loaded------------')
	load_model_end = time.localtime(time.time())
	print_time('load model',load_model_start, load_model_end)
	#set optimizer
	lr = args.learning_rate
	optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
	lr_scheduled = lr_scheduler.MultiStepLR(optimizer, [300, 1000])
	steps = 0
	
	#train
	while steps < args.max_steps:
		print('Step {}/{}'.format(steps, args.max_steps))
		print('*'*20)
		
		for phase in ['train', 'valid']:
			if phase == 'train':
				i3d.train()
			else:
				print('validing...')
				i3d.eval()
			
			#losses = AverageMeter()
			#accuracies = AverageMeter()
			losses = 0.0
			accuracies = 0.0
			total = 0
			for data in dataloaders[phase]:
				#input
				inputs, labels = data['video_x'], data['video_label']
				if bool(args.use_cuda):
					inputs = Variable(inputs.float()).cuda()
					labels = Variable(labels.float()).cuda()
				else:
					inputs = Variable(inputs.float())
					labels = Variable(labels.float())
				
				
				#print(inputs[:,0,0,0,:])
				#outputs
				outputs, out_logits = i3d(inputs)
				_, predicted = torch.max(out_logits, 1)
				#print('predicted',predicted)#shape of (N,1)
				#print('out_puts:',outputs)#shape of(N,C)
				#print('labels:',labels)#shape of (N,1)
				loss = F.cross_entropy(out_logits, labels.long().squeeze())
				#acc = calculate_accuracy(out_logits ,labels)
				
				# update loss and accuracy
				#losses.update(loss.data[0], inputs.size(0))
				#accuracies.update(acc, inputs.size(0))
				losses += float(loss)
				total +=labels.size(0)
				accuracies += float((predicted == labels.long().squeeze()).sum())
				#backward
				optimizer.zero_grad()
				if phase == 'train':
					steps += 1
					loss.backward()
					optimizer.step()
					lr_scheduled.step()
					if steps % args.print_step == 0:
						step_time = '%s'%datetime.now()
						time1 = step_time.split(' ')[0]
						time2 = step_time.split(' ')[1].split('.')[0]
						strtime = '%s %s'%(time1, time2)
						print('{} [{}] steps:{} loss:{:4f} accruacy:{:3f}'.
							 format(strtime, phase, steps, losses/total, accuracies/total))
					if steps % 1000 == 0:
						torch.save(i3d.state_dict(), 'model/ucf/model_ucf_'+str(steps))
			if phase == 'valid':
				print('{} [{}]  loss:{:4f} accruacy{:3f}'.
							 format(strtime, phase, losses/total, accuracies/total))
	
if __name__ == '__main__':
	# parse argument
	parser = argparse.ArgumentParser('Train UCF101 on i3d modle') 
	parser.add_argument('--mode', type=str, default='rgb', help='modality')
	parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
	parser.add_argument('--num_workers', type=int, default=4, help='num_worker')
	parser.add_argument('--learning_rate',type=float, default=0.1, help='learning_rate')
	parser.add_argument('--max_steps',type=int, default=8000, help='max_steps')
	parser.add_argument('--num_classes',type=int, default=101, help='number of action class')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
	parser.add_argument('--use_cuda', type=int, default=1, help='use cuda')
	parser.add_argument('--list_path', type=str, default='data/UCF101_dataset/ucfTrainTestlist/new_list', help='the path to list')
	parser.add_argument('--root_dir', type=str, default='data/UCF101_dataset/UCF_FRAME', help='the root directory of data')
	parser.add_argument('--print_step', type=int, default=50)
	args = parser.parse_args()

	train(args)
	
	
	
	
	
	
	
