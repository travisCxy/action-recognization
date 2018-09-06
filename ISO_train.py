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

# torchvision
from torchvision import transforms
from data.ssd_dataset.dataloader import Iso_GD,get_dataloaders
from data.ssd_dataset.transform_utils import *

import numpy as np
from i3d_net import I3D,Unit3Dpy,weight_init
from utils.print_time import print_time
from utils.write_log import write_log
from datetime import datetime
from tensorboardX import SummaryWriter

#train_log_file = open('train_log_file.txt','w+')
model_save_log = open('model/model_save_log.txt','w+')
def train(args):
    writer = SummaryWriter()
    #Prepare dataset
    dataloaders = get_dataloaders(args)
    print('------------ data  Loaded  ------------')
    # model
    load_model_start = time.localtime(time.time())
    i3d = I3D(num_classes=args.num_classes, modality=args.mode, dropout_prob=args.dropout_prob)
    if args.mode == 'rgb':
        i3d.load_state_dict(torch.load(args.model_path))
    elif args.mode == 'flow':
        #i3d.load_state_dict(torch.load('model/model_flow.pth'))
        i3d.apply(weight_init)
    else:
        raise ValueError('mode excepted to be [rgb|flow],but get{}'.format(mode))
    #i3d.conv3d_0c_1x1 = Unit3Dpy(1024, args.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), activation=None, use_bias=True, use_bn=False)
    if bool(args.use_cuda):
        i3d.cuda()
    print('------------ model loaded -------------')
    load_model_end = time.localtime(time.time())
    print_time('load model',load_model_start, load_model_end)
    # set optimizer
    lr = args.learning_rate
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduled = lr_scheduler.StepLR(optimizer, 10000)
    steps = 0
    epoch = 1
    #train
    while steps < args.max_steps:
        print('Step {}/{}'.format(steps, args.max_steps))
        print('-'*10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                i3d.train()
            else:
                print('*********validing*********')			
                i3d.eval()

            running_loss = 0.0
            running_corrects = 0.0
            total = 0
            best_acc = 0.0          # Iterating
            for data in dataloaders[phase]:
                # input
                inputs, labels = data['video_x'], data['video_label']
                if bool(args.use_cuda):
                    inputs = Variable(inputs.float()).cuda()
                    labels = Variable(labels.float()).cuda()
                else:
                    inputs = Variable(inputs.float())
                    labels = Variable(labels.float())

                # zero the parameter gradients
                optimizer.zero_grad()
                # output
                _, out_logits = i3d(inputs)
                _, predicted = torch.max(out_logits, 1)
               # print(predicted)
               # write_log(train_log_file,predicted,labels,steps)
                loss = F.cross_entropy(out_logits, labels.long().squeeze())

                # loss and accuracy
                running_loss += float(loss)
                total += labels.size(0)
                running_corrects += float((predicted == labels.long().squeeze()).sum())

                if phase == 'train':
                    steps += 1
                    loss.backward()
                    optimizer.step()
                    lr_scheduled.step()
                    if steps % 50 == 0:
                        step_time = '%s'%datetime.now()
                        time1 = step_time.split(' ')[0]
                        time2 = step_time.split(' ')[1].split('.')[0]
                        strtime = '%s %s'%(time1, time2)
                        print('{} [{}] steps:{}  loss:{:4f}  accuracy:{:3f}'
                                   .format(strtime, phase, steps, running_loss/total, running_corrects/total))
                    writer.add_scalar('data/loss',running_loss/total,steps)
                    writer.add_scalar('data/acc',running_corrects/total,steps)
                       # torch.save(i3d.state_dict(),'model/iso_model/'+str(time1)+'/'+'epoch'+str(epoch))
                       # epoch += 1
                       # model_save_log.write('model save at model/iso_model/'+str(time1)+'/'+'epoch'+str(epoch)+
                       # '   valid loss:'+str(running_loss/total)+'   valid accuracy:'+str(running_corrects/total)+
                       # '   dropout_prob:'+str(args.dropout_prob)+'   weight_decay:'+str(args.weight_decay))

            if phase == 'valid':
                step_time = '%s'%datetime.now()
                time1 = step_time.split(' ')[0]
                time2 = step_time.split(' ')[1].split('.')[0]
                strtime = '%s %s'%(time1, time2)
                print('{} steps:{} loss:{:4f} accuracy:{:4f}'
                      .format(phase, steps, running_loss/total, running_corrects/total))
                torch.save(i3d.state_dict(),'model/iso_model/'+str(time1)+'/'+str(args.note)+'epoch'+str(epoch)+'{:3f}'.format(running_corrects/total))
                epoch += 1
                model_save_log.write(str(strtime)+'model save at model/iso_model/'+str(time1)+'/'+'epoch'+str(epoch)+'\n'
				'	valid loss:'+str(running_loss/total)+'   valid accuracy:'+str(running_corrects/total)+
				'   dropout_prob:'+str(args.dropout_prob)+'   weight_decay:'+str(args.weight_decay)+'\n')
    writer.close()

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser('Train Iso_GD on i3d modle') 
    parser.add_argument('--mode', type=str, default='rgb', help='modality')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_worker')
    parser.add_argument('--learning_rate',type=float, default=0.01, help='learning_rate')
    parser.add_argument('--max_steps',type=int, default=45000, help='max_steps')
    parser.add_argument('--num_classes',type=int, default=249, help='number of action class')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='weight_decay')
    parser.add_argument('--use_cuda', type=int, default=1, help='use cuda')
    parser.add_argument('--list_path', type=str, default='data/ssd_dataset/list', help='the path to list')
    parser.add_argument('--root_dir', type=str, default='data/ssd_dataset', help='the root directory of data')
    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--dropout_prob', type=float, default=0.5)
    parser.add_argument('--model_path',type=str,default='model/iso_model/2018-08-04/rgbbest0.595')
    parser.add_argument('--note',type=str)
    args = parser.parse_args()

    train(args)

		

		
		
