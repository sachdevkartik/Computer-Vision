#!/bin/python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

#custom import
from models.cnn import CalcConvFormula, CalcConvOutShape
import numpy as np
import time
import os

# # 3x3 convolution
# #TODO remove and import from cnn.py file
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)
#
# #TODO remove and import from cnn.py file
# def CalcConvFormula(W, K, P, S):
#     return int(np.floor(((W - K + 2 * P) / S) + 1))
#
# #TODO remove and import from cnn.py file
# # https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
# # Calculate the output shape after applying a convolution
# def CalcConvOutShape(in_shape, kernel_size, padding, stride, out_filters):
#     # Multiple options for different kernel shapes
#     if type(kernel_size) == int:
#         out_shape = [CalcConvFormula(in_shape[i], kernel_size, padding, stride) for i in range(2)]
#     else:
#         out_shape = [CalcConvFormula(in_shape[i], kernel_size[i], padding, stride) for i in range(2)]
#
#     return (out_shape[0], out_shape[1], out_filters)  # , batch_size... but not necessary.

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, num_features, use_batch_norm=False):
        super(ResBlock, self).__init__()
        self.num_features = num_features
        self.conv_layer1 = nn.Conv2d(num_features, num_features,  kernel_size=3, stride=1, padding=1)
        self.relu_layer = nn.ReLU()
        self.conv_layer2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm_layer1 = nn.BatchNorm2d(self.num_features)
            self.batch_norm_layer2 = nn.BatchNorm2d(self.num_features)

    def forward(self, x):
        residual = x
        x = self.conv_layer1(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer1(x)    #nn.BatchNorm2d(self.num_features)

        x = self.relu_layer(x)

        x = self.conv_layer2(x)
        if self.use_batch_norm:
            x = self.batch_norm_layer2(x)

        x += residual
        x = self.relu_layer(x)
        return x


# def make_conv_relu(kernel_size=3, padding=1):
#     return nn.ReLU()




# ResNet
# num_in_channel =  testloader.dataset.data.shape[3]
class ResNet(nn.Module):
    def __init__(self, in_features, num_class, feature_channel_list, batch_norm= False, num_stacks=1):
        super(ResNet, self).__init__()
        self.in_features = in_features
        self.num_in_channel = in_features[2]
        self.num_class = num_class
        self.feature_channel_list = feature_channel_list
        self.num_residual_blocks = len(self.feature_channel_list)
        self.num_stacks = num_stacks
        self.batch_norm = batch_norm
        # self.block = ResBlock()

        self.shape_list = []
        self.shape_list.append(in_features)

        self.module_list = nn.ModuleList()
        self.build_()

    def GetCurShape(self):
        return self.shape_list[-1] #TODO define


    def build_(self):
        #track filter shape
        cur_shape = self.GetCurShape()
        cur_shape = CalcConvOutShape(cur_shape, kernel_size=7, padding=1, stride=2, out_filters= self.feature_channel_list[0])
        self.shape_list.append(cur_shape)

        if len(self.in_features) == 2:
            in_channels = 1
        else:
            in_channels = self.in_features[2]

        # First Conv layer 7x7 stride=2, pad =1
        self.module_list.append(nn.Conv2d(in_channels= in_channels,
                                    out_channels= self.feature_channel_list[0],
                                    kernel_size=7,
                                    stride=2,
                                    padding=3))


        #batch norm
        if self.batch_norm: #batch_norm
            self.module_list.append(nn.BatchNorm2d(self.feature_channel_list[0]))

        # ReLU()
        self.module_list.append(nn.ReLU())

        for i in range(self.num_residual_blocks-1):
            in_size = self.feature_channel_list[i]
            out_size = self.feature_channel_list[i+1]

            res_block = ResBlock(in_size, use_batch_norm=True)
            #Stacking Residual blocks
            for num in range(self.num_stacks):
                self.module_list.append(res_block)

            # Intermediate Conv and ReLU()
            self.module_list.append(nn.Conv2d(in_channels=in_size,
                                              out_channels= out_size,
                                              kernel_size=3,
                                              padding=1,
                                              stride=2))
            # track filter shape
            cur_shape = CalcConvOutShape(cur_shape, kernel_size=3, padding=1,
                                         stride=2, out_filters=out_size)
            self.shape_list.append(cur_shape)

            # batch norm
            if self.batch_norm:  # batch_norm
                self.module_list.append(nn.BatchNorm2d(out_size))

            self.module_list.append(nn.ReLU())

        #TODO include in the main loop
        #Last Residual block
        res_block = ResBlock(out_size, use_batch_norm=True)
        for num in range(self.num_stacks):
            self.module_list.append(res_block)

        #Last AvgPool layer
        self.module_list.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        # track filter shape
        cur_shape = CalcConvOutShape(cur_shape, kernel_size=2, padding=0, stride=2, out_filters=out_size)
        self.shape_list.append(cur_shape)

    def AddMLP(self, MLP):
        if MLP:
            self.module_list.append(MLP)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        x = x.view(x.size(0), -1)  # flat #TODO check if it works
        return x

    def SaveModel(self, file):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), file)



        # self.build_()
        # def build_(self):
        #     for i in range(0,1):
        #         self.module_list.append(self.conv_layer)
        #
        #         if self.use_batch_norm:
        #             self.module_list.append(nn.BatchNorm2d(self.num_features))
        #
        #         self.module_list.append(self.relu_layer)
        #
        #         if self.use_dropout:
        #             self.module_list.append(nn.Dropout(p=0.15))


        # residual = x
        # for module in self.module_list:
        #     out = module(x)
        # out +=residual
        # out = self.relu_layer(out)
        # return out


