import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math 
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
	# 3x3x3 convolution with padding
	return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
	# downsample x and pad in channels dim
	out = F.avgpool3d(x, kernel_size=1, stride=stride)
	zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
							 out.size(2),out.size(3),out.size(4)).zero_()
	if isintance(out.data, torch.cuda.FloatTensor()):
		zero_pads = zero_pads.cuda()
	out = Variable(torch.cat([out.data, zero_pads], dim=1))
	return out

class BasicBlock(nn.Module):
	# ResNet basic block
	expansion = 1
	
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock,self).__init__()
		self.conv1 = conv3x3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorn3d(planes)
		self.downsample = downsample
		
	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		
		if not self.downsample:
			residual = downsample(residual)
		out = out + residual
		out = self.relu(out)
		
		return out

class Bottleneck(nn.Module):
	expansion = 4
	
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		
	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		
		out = self.conv3(out)
		out = self.bn3(out)
		
		if self.downsample is not None:
			residual = self.downsample(x)
		
		out = out + residual
		out = self.relu(out)
		
		return out 

	
class Encoder(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Encoder, self).__init__()
		
		self.conv1 = nn.Conv3d(in_channels, 64, (3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
		self.bn1 = nn.BatchNorm3d(64)
		self.relu1 = nn.relu(inplace=True)
		
		self.conv2 = nn.Conv3d(64, out_channels, (1, 3, 3), stride=1, padding=(0, 1, 1))
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.relu2 = nn.relu(inplace=True)
		
		self.conv3 = nn.Conv3d(out_channesl, out_channels, (4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1))
		self.bn3 = nn.BatchNorm3d(out_channels)
		self.relu3 = nn.relu(inplace=True)
		
		self.maxpool3d = nn.MaxPool3d((3, 3, 3), stride=2, padding=1)
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		
		out = self.conv3(out)
		out = self.bn3(out)
		out = self.relu3(out)
		
		out = self.maxpool3d(out)
		
		return out
		
		
		
		
		
		
class ResNet_Encoder(nn.Module):
	def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.Encoder = Encoder(3, 64)
		self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
		self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], shortcut_tpe, stride=2)
		last_duration = int(math.ceil(sample_duration/16))
		last_size = int(math.ceil(sample_size / 32))
		self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
		self.conv = nn.Conv1d(512*block.expansion, num_classes, kernel_size =1)
		
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		
	def _make_layer(slef, block, planes, blocks, shortcut_type, stride=1):
		'''
		Argument
		block : Type of block, 'BasicBlock' or 'Bottleneck'
		planes : out channels
		blocks : number of blocks
		shortcut_type : Type of downsample 'AvgPool+Pad' or 'conv1x1x1'
		stride : stride
		'''
		downsample =None
		if stride !=1 or self.inplanes != planes*block.expansion:
			if shortcut_type = 'A':
				downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride)
			else:
				downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes*block.expansion,kernel_size=1, 											   					stride=stride, bias=False), nn.BatchNorm3d(planes*block.expansion))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		
		x = self.Encoder(x)
				
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.conv(x)
		
		return x

	
	def resnet10(**kwargs):
		''' Constructs a 3D ResNet-10 model
		'''
		model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
		return  model
	
	def resnet18(**kwargs):
		''' Constructs a 3D ResNet-18 model
		'''
		model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
		return model

	def resnet34(**kwargs):
		''' Constructs a 3D ResNet-34 model
		'''
		model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
		return model

	def resnet50(**kwargs):
		''' Constructs a 3D ResNet-50 model
		'''
		model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
		return model	

	def resnet101(**kwargs):
		''' Constructs a 3D ResNet-101 model
		'''
		model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
		return model
	
	def resnet152(**kwargs):
		''' Constructs a 3D ResNet-152 model
		'''
		model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
		return model	
	
	def resnet200(**kwargs):
		''' Constructs a 3D ResNet-200 model
		'''
		model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
		return model
		
		
		
		
		
		
		
		
		
		
		
		
		
