import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import ReplicationPad3d
from net.i3d_net import I3D
from torch.autograd import Function

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom
    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    """padding_shape from(depth_pad_shape,height_pad_shape,width_pad_shape) 
       transform to(height_pad_shape,width_pad_shape,depth_pad_shape)
    """
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad !=padding_init:
            all_same = False
    return all_same, padding_init


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == 'VALID':
        return [0, 0]
    elif padding_name == 'SAME':
        return [math.floor(int(conv_shape[0]) / 2),
                math.floor(int(conv_shape[1]) / 2),
                math.floor(int(conv_shape[2]) / 2)]
    else:
        raise ValueError('Invalid padding name' + padding_name)



class Unit3Dpy(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1),
                 stride=(1, 1, 1), activation='relu', padding='SAME',
                 use_bias=False, use_bn=True):
        super(Unit3Dpy, self).__init__()
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError('padding should be in [VALID|SAME] but'
                             'got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:   # which mean pads in different dims are not same
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                   in_channels, out_channels, kernel_size, stride=stride, bias=use_bias)
            # nn.Conv3d()中的pad要不是int要不是3维tuple，所以要提前添加pad3d函数
            else:
                self.conv3d = torch.nn.Conv3d(in_channels,
                    out_channels, kernel_size, stride = stride, padding=pad_size, bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(in_channels,
            out_channels, kernel_size, padding=padding_shape, stride=stride, bias=use_bias)
        else:
            raise ValueError('padding should be in [VALID|SAME] but'
                             'got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3DTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3DTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch 3
        branch_3_pool = MaxPool3DTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out 
	
#create a self-defined layer
class Fusion(Function):
    @staticmethod
    def forward(ctx, weight, input, input_requires_grad=True):
        ctx.input_requires_grad = input_requires_grad
        ctx.save_for_backward(weight, input)
        input = input.float()
        output = torch.Tensor(input.size(0),input.size(1),input.size(1))
        batch_size = input.size(0)
        for i in range(batch_size):
            output[i] = torch.mm(input[i] ,weight[i])
        return output
    @staticmethod
    def backward(ctx, grad_output):
        batch_size = grad_output.size(0)
        weight, input = ctx.saved_variables
        grad_weight = torch.Tensor(weight.shape)
        grad_input = torch.Tensor(input.shape)
        for i in range(batch_size):
            grad_weight[i] = (grad_output[i].mm(input[i].cpu())).t()
        if ctx.input_requires_grad:
            for i in range(batch_size):
                grad_input[i] = grad_output[i].mm(weight[i].t().cpu())
        else:
            grad_input = None
        return grad_weight.cuda(), grad_input.cuda(), None  #设置cuda

#create a Module
class TimeFusion(nn.Module):
    def __init__(self, input_shape, output_shape, bias=None):
        super(TimeFusion, self).__init__()
        B = output_shape[0]
        T = input_shape[1]
        C = output_shape[1]
        self.weight = nn.Parameter(torch.Tensor(B, T, C))#.cuda()
        #print(self.weight)
		#if bias:
         #   self.bias = nn.Parameter(torch.Tensor(output_shape))
        #else:
        #    self.register_parameter('bias', None)
        self.weight.data.uniform_(-1,1)
        #if bias is not None:
          #  self.bias.data.uniform_(-1,1)
    def forward(self, input):
        return Fusion.apply(self.weight, input)

def get_score(x):
    assert x.size(1) == x.size(2)
    batch_size, h, w = x.size(0), x.size(1), x.size(2)
    score = torch.Tensor(batch_size, h)
    x = x.float()
    for b in range(batch_size):
         for i in range(h):
             score[b,i] = x[b,i,i]
    return score

#create a encoder
class encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, branch=None):
        super(encoder, self).__init__()
        #layer1:conv-bn-relu-pool
	#encode information between joints for every part
	#shape from(3,768,56,56)-->(64,256,28,28) 
        self.conv3d_3x3x3 = torch.nn.Conv3d(in_channels, 64, (3,3,3), stride=(3,1,1),padding=(0,1,1))
        self.bn1 = torch.nn.BatchNorm3d(64)
        self.relu1 = torch.nn.functional.relu
        self.maxpool3d_1 = MaxPool3DTFPadding((1,3,3), (1,2,2), padding='SAME')
	#layer2:conv-bn-relu
        self.conv3d_1x3x3 = torch.nn.Conv3d(64, 64, (1,3,3), stride=(1,1,1), padding=(0,1,1))
        self.bn2 = torch.nn.BatchNorm3d(64)
        self.relu2 = torch.nn.functional.relu
        #layer3:conv-bn-relu
        self.conv3d_3_1_2x3x3 = torch.nn.Conv3d(64, out_channels, (2,3,3), stride=(2,1,1),padding=(0,1,1))
        self.bn3_1 = torch.nn.BatchNorm3d(out_channels)
        self.relu3_1 = torch.nn.functional.relu
        self.conv3d_3_2_2x3x3 = torch.nn.Conv3d(192, out_channels, (2,3,3), stride=(2,1,1),padding=(0,1,1))
        self.bn3_2 = torch.nn.BatchNorm3d(out_channels)
        self.relu3_2 = torch.nn.functional.relu
    def forward(self,x):
        out = self.conv3d_3x3x3(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool3d_1(out)
        out = self.conv3d_1x3x3(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3d_3_1_2x3x3(out)
        out = self.bn3_1(out)
        out = self.relu3_1(out)
        out = self.conv3d_3_2_2x3x3(out)
        out = self.bn3_2(out)
        out = self.relu3_2(out)
        return out

#Net
class I3D_with_encoder(torch.nn.Module):
    def __init__(self, num_classes=400, modality='rgb', dropout_prob=0.5,  name='inception', fusion_type='mean', batch_size=8):
        super(I3D_with_encoder, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 3
        else:
            raise ValueError('modality excepted to be [rgb|flow],but get {}'.format(modality))
        self.modality = modality
	    #encoder 
        self.encoder = encoder(3, 192)
        '''
	    conv3d_1a_7x7 = Unit3Dpy(out_channels=64, in_channels=in_channels,
                                 kernel_size=(7, 7, 7), stride=(2, 2, 2), padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3DTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # 2rd conv-conv-pool
        conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1), padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1), padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3DTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        '''
        # 3th Mixed-Mixed-pool
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3DTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # 4th Mixed-Mixed-Mixed-Mixed-Mixed-pool
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        self.maxPool3d_5a_2x2 = MaxPool3DTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # 5th Mixed
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        if fusion_type != 'conv':
            conv3d_0c_1x1_ = Unit3Dpy(in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                activation=None, use_bias=True, use_bn=False)
            self.conv3d_0c_1x1_ = conv3d_0c_1x1_
        else:
            fusion_conv = torch.nn.Conv3d(1024, self.num_classes, kernel_size=(7, 1, 1))
            self.fusion_conv = fusion_conv
        if fusion_type == 'avgpool':
            self.time_avg_pool = torch.nn.AvgPool1d(7,1)
        elif fusion_type == 'maxpool':
            self.time_max_pool = torch.mm.MaxPool1d(7,1)
        self.fusion = TimeFusion([49,7],[batch_size,49,49])      
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        """
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        """
        out = self.encoder(inp)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        if self.fusion_type != 'conv':
              out = self.conv3d_0c_1x1_(out)
        else:
            out = self.fusion_conv(out)
        out = out.squeeze()    # shape:(N,C,D)
        if self.fusion_type == 'mean':
            out = out.mean(2)       # shape:(batch_size,num_classes)
        elif self.fusion_type == 'avgpool':
            out = self.time_avg_pool(out)
            out = out.squeeze(2)
        elif self.fusion_type == 'maxpool':
            out = self.time_max_pool(out)
            out = out.squeeze(2)
        elif self.fusion_type == 'weight':
            out = self.fusion(out)
            out = get_score(out)
        elif self.fusion_type == 'conv':
            pass
        else:
            raise ValueError('Got a wrong value for fusion_type'.format(self.fusion_type))
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits


		
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('conv3d') != -1:
        model.weight.data.xavier_normal()
    elif classname.find('batch3d') != -1:
        model.weight.data.xavier_normal()
        model.bias.data.fill_(0)

