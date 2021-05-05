#!/bin/python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP, DefaultCifar10MLP

from utils.utils import GetActivation

import os


# Use the formulat:
#    [(W-K+2P)/S] + 1
# 	where:
#		W:  Is the input volume size for each dimension
#		K:  Is the kernel size
#		P:  Is the padding
#		S:  Is the stride
def CalcConvFormula(W, K, P, S):
	return int(np.floor(((W - K + 2*P)/S) + 1))

# https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
# Calculate the output shape after applying a convolution
def CalcConvOutShape(in_shape, kernel_size, padding, stride, out_filters):
	# Multiple options for different kernel shapes
	if type(kernel_size) == int:
		out_shape = [CalcConvFormula(in_shape[i], kernel_size, padding, stride) for i in range(2)]
	else:
		out_shape = [CalcConvFormula(in_shape[i], kernel_size[i], padding, stride) for i in range(2)]
		
	return (out_shape[0], out_shape[1], out_filters) # , batch_size... but not necessary.


class CNN(nn.Module):
	def __init__(self
			, in_features
            , out_features
            , conv_filters
            , conv_kernel_size
            , conv_strides
            , conv_pad
            , actv_func
            , max_pool_kernels
            , max_pool_strides
            , MLP=None
            , pre_module_list=None
            , use_dropout=False
            , use_batch_norm=False
            , device="cpu"
            ):
		super(CNN, self).__init__()

		# Gerneral model Properties
		self.in_features = in_features
		self.out_features = out_features

		# Convolution operations
		self.conv_filters = conv_filters
		self.conv_kernel_size = conv_kernel_size
		self.conv_strides = conv_strides
		self.conv_pad = conv_pad

		# Convolution Activiations
		self.actv_func = actv_func

		# Max Pools
		self.max_pool_kernels = max_pool_kernels
		self.max_pool_strides = max_pool_strides

		# Regularization
		self.use_dropout = use_dropout
		self.use_batch_norm = use_batch_norm

		# Number of conv/pool/act/batch_norm/dropout layers we add
		self.n_conv_layers = len(self.conv_filters)

		# Create the module list
		if pre_module_list:
			self.module_list = pre_module_list
		else:
			self.module_list = nn.ModuleList()

		self.shape_list = []
		self.shape_list.append(self.in_features)


		self.build_()

		# Re-Build MLP Network.
		# Connects the output of the convs to 'out_features'
		self.AddMLP(MLP)

		# Send to gpu
		self.device = device
		self.to(self.device)

	def AddMLP(self, MLP):
		if MLP:
			self.module_list.append( MLP )

	def build_(self):
		# Track shape
		cur_shape = self.GetCurShape()

		for i in range(self.n_conv_layers):
			if i == 0:
				if len(self.in_features) == 2:
					in_channels = 1
				else:
					in_channels = self.in_features[2]
			else:
				in_channels = self.conv_filters[i-1]

			cur_shape = CalcConvOutShape(cur_shape, self.conv_kernel_size[i], self.conv_pad[i], self.conv_strides[i], self.conv_filters[i])
			self.shape_list.append(cur_shape)

			conv = nn.Conv2d( 	in_channels = in_channels,
								out_channels = self.conv_filters[i],
								kernel_size = self.conv_kernel_size[i],
								padding=self.conv_pad[i],
								stride = self.conv_strides[i]
								 )
			self.module_list.append( conv )

			if self.use_batch_norm:
				self.module_list.append(nn.BatchNorm2d(cur_shape[2]))

			if self.use_dropout:
				self.module_list.append(nn.Dropout(p=0.15))

			# Add the Activation function
			if self.actv_func[i]:
				self.module_list.append( GetActivation(name=self.actv_func[i]) )

			if self.max_pool_kernels:
				if self.max_pool_kernels[i]:
					self.module_list.append( nn.MaxPool2d(self.max_pool_kernels[i], stride=self.max_pool_strides[i]) )
					cur_shape = CalcConvOutShape(cur_shape, self.max_pool_kernels[i], 0, self.max_pool_strides[i], cur_shape[2])
					self.shape_list.append(cur_shape)
					

	def forward(self, x):
		for module in self.module_list:
			x = module(x)

		return x

	def GetCurShape(self):
		return self.shape_list[-1]

	def SaveModel(self, file):
		directory = os.path.dirname(file)
		if not os.path.exists(directory):
			os.makedirs(directory)
		torch.save(self.state_dict(), file)

	def RestoreModel(self, file):
		self.load_state_dict(torch.load(file))




def DefaultCifar10CNN(device):
	cnn = CNN( in_features=(32,32,3),
				out_features=10,
				conv_filters=[32,16],
				conv_kernel_size=[5,5],
				conv_strides=[1,1],
				conv_pad=[0,0],
				max_pool_kernels=[(2,2), (2,2)],
				max_pool_strides=[2,2],
				use_dropout=False,
            	use_batch_norm=False,
				actv_func=["leakyrelu","leakyrelu"],
				device=device
		)
	# Create MLP
	# Calculate the input shape
	s = cnn.GetCurShape()
	in_features = s[0]*s[1]*s[2]

	mlp = MLP(in_features, 
                10,
                [120,84],
                ["leakyrelu", "leakyrelu"],
                use_batch_norm=False,
                use_dropout=False,
                use_softmax=False,
                device=device)

	# mlp = DefaultCifar10MLP(device=device, in_features=in_features)

	cnn.AddMLP(mlp)

	return cnn

