import os 
import sys
import argparse
import time
import numpy as np
from datetime import datetime

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
#data
from data.ntu_dataset.dataloader import get_dataloaders
from net.i3d_net_encoder import I3D_with_encoder
from utils.print_time import print_time, get_strtime
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
def run(args):
	if args.work == 'train':
		writer =  SummaryWriter()
		#data
		dataloaders = get_dataloaders(args)
		print('*'*15+'data loaded'+'*'*15)
		#model
		I3D = I3D_with_encoder(num_classes=args.num_classes, fusion_type=args.fusion_type, batch_size=args.batch_size)
		model_dict = I3D.state_dict()
		kinetic_dict = torch.load(args.model_path)
		kinetic_dict = {k:v for k,v in kinetic_dict.items() if k in model_dict}
		model_dict.update(kinetic_dict)
		I3D.load_state_dict(model_dict)
		if args.use_cuda:
			I3D.cuda().half()
		print('*'*15+'model loaded'+'*'*15)
		#optimizer
		lr = args.learning_rate
		if args.optim == 'SGD':
			optimizer = optim.SGD(I3D.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.optim == 'Adam':
			torch.nn.utils.clip_grad_norm_(I3D.parameters(),0.1)
			optimizer = optim.Adam(I3D.parameters(), lr=lr, weight_decay=args.weight_decay)
		#lr_sheduled = lr_scheduler.MultiStepLR(optimizer, [])
		epoch = 1
		steps = 0
		train_loss = 0.0
		train_corrects = 0.0
		train_steps = 0
		while epoch<args.num_epoch:
			print('*'*40)
			#train_loss = 0.0
			#train_corrects = 0.0
			valid_loss = 0.0
			valid_corrects = 0.0
			valid_steps = 0
			iteration = 0
			log = open('log/ntu_model.txt', 'a+')
			#train
			'''
			for data in dataloaders['train']:
				#input
				I3D.train()
				inputs, labels = data['video'], data['label']
				if bool(args.use_cuda):
					inputs = Variable(inputs.half().cuda())
					labels = Variable(labels.half().cuda())
				else:
					inputs = Variable(inputs.half())
					labels = Variable(labels.half())
				#output
				out, out_logits = I3D(inputs)
#				print(out_logits.size())
				_, predicted = torch.max(out_logits, 1)
				#loss and acc
				optimizer.zero_grad()
				loss = (F.cross_entropy(out_logits, labels.long().squeeze())).half()
				train_loss += loss.data[0]
				train_corrects += ((predicted.short() == labels.short().squeeze()).sum()).half()
				iteration += labels.size(0)
				train_steps += labels.size(0)
				steps += 1
				loss.backward()
				optimizer.step()
				#lr_sheduled.step()
				if steps % args.print_step == 0:
					time = get_strtime()
					print('Training  '+str(time)+'epoch:'+str(epoch)+'  iterations:'+str(iteration)+
						'  loss:%3f'%(train_loss/train_steps)+'  accuracy:%3f'%(train_corrects/train_steps))
					writer.add_scalar('data/loss', float(train_loss/(train_steps)), train_steps)
					writer.add_scalar('data/acc', float(train_corrects/(train_steps)), train_steps)
			time = '%s'%datetime.now()
			time1 = time.split(' ')[0]
			torch.save(I3D.state_dict(), 'model/ntu_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
			print('model has been saved at model/ntu_model/'+str(time1)+'/'+str(args.note)+'_'+str(epoch))
			log.write('Note:'+str(args.note)+' epoch:'+str(epoch)+' train_acc:%3f'%(train_corrects/train_steps))
'''
			print('validing')
			#valid	
			for data in dataloaders['train']:
				I3D.eval()
				inputs, labels = data['video'], data['label']
				if bool(args.use_cuda):
					inputs = Variable(inputs.half().cuda())
					labels = Variable(labels.half().cuda())
				else:
					inputs = Variable(inputs.half())
					labels = Variable(labels.half())
				out, out_logits = I3D(inputs)
				_, predicted = torch.max(out, 1)
				print('predicted', predicted)
				#print('input1', inputs[0])
				print('labels', labels)
				loss = (F.cross_entropy(out_logits, labels.long().squeeze())).half()
				valid_loss += loss.data[0]
				valid_steps += labels.size(0)
				valid_corrects += ((predicted == labels.long().squeeze()).sum()).half()
				#print(valid_steps)
			print('Valid {} epoch:{} loss:{:4f} accuracy{:3f}'
					 .format(get_strtime(), epoch, valid_loss/valid_steps, valid_corrects/valid_steps))
			log.write(' valid_acc:%3f'%(valid_corrects/valid_steps)+'\n')
			epoch += 1
	writer.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser('TRAIN NTU')				
	parser.add_argument('--work', type=str, default='train', help='indicate which work doing')			
	parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
	parser.add_argument('--valid_batch_size', type=int, default=4, help='batch_size')
	parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
	parser.add_argument('--learning_rate', type=float, default=0.1, help='learning_rate')
	parser.add_argument('--num_classes', type=int, default=49, help='num of action classes')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
	parser.add_argument('--use_cuda', type=int, default=1, help='use cuda')
	parser.add_argument('--list_path', type=str, default='data/ntu_dataset/list', help='the path to list')
	parser.add_argument('--root_dir', type=str, default='data/ntu_dataset/ntu_frame', help='the root directory of data')
	parser.add_argument('--print_step', type=int, default=50, help='step to print infomation')
	parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout prob of network')			
	parser.add_argument('--model_path',type=str,help='path to pretrained model')			
	parser.add_argument('--note',type=str,help='note to indicate which train ')		
	parser.add_argument('--num_epoch',type=int, default=10,help='iteration epoch')			
	parser.add_argument('--skeleton_info_path', type=str, default='data/ntu_dataset/nturgb+d_skeletons')
	parser.add_argument('--fusion_type', type=str, default='mean', help='The method to fusion time dimension')
	parser.add_argument('--optim', type=str, default='SGD')
	args = parser.parse_args()
	run(args)			
				
				
				
				
				
				
				
				
				
				
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
