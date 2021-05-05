#!/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import GetActivation

import os


class MLP(nn.Module):
    def __init__(self
            , in_features
            , out_features
            , hidden_layers
            , actv_func
            , pre_module_list=None
            , use_dropout=False
            , use_batch_norm=False
            , use_softmax=True
            , device="cpu"
            ):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.actv_func = actv_func
        self.use_softmax = use_softmax

        self.device = device

        # Add on to another model
        if pre_module_list:
            self.module_list = pre_module_list
        else:
            self.module_list = nn.ModuleList()


        self.build_()

        # Send to gpu
        self.to(self.device)

    def build_(self):
        # Activation Functions for Fully connected layers #
        # Start with input dimensions
        dim = self.in_features
        for i in range(self.num_hidden_layers):
            # Create a fully connected layer between the last layer
            #  and the current hidden layer
            self.module_list.append(nn.Linear(dim, self.hidden_layers[i]))
            # Update the current dimension
            dim = self.hidden_layers[i]

            if self.use_batch_norm:
                self.module_list.append( nn.BatchNorm1d(dim, affine=True) )

            # Add the Activation function
            self.module_list.append( GetActivation(name=self.actv_func[i]) )

            if self.use_batch_norm:
                self.module_list.append( nn.BatchNorm1d(dim) )

            if self.use_dropout:
                self.module_list.append( nn.Dropout(p=0.15) )

        # Fully connect to output dimensions
        if dim != self.out_features:
            self.module_list.append( nn.Linear(dim, self.out_features) )


    def forward(self, x):
        # Flatten the 2d image into 1d
        # Also convert into float for FC layer
        x = torch.flatten(x.float(), start_dim=1)

        # Apply each layer in the module list
        for i in range( len(self.module_list) ):
            x = self.module_list[i](x)

        # Optional, use softmax
        if self.use_softmax:
            x = F.softmax(x, dim=1)

        return x

    def SaveModel(self, file):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), file)

    def RestoreModel(self, file):
        self.load_state_dict(torch.load(file))

    # https://discuss.pytorch.org/t/reset-model-weights/19180/4
    def Reset(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()





def DefaultCifar10MLP(device="cpu", act_func="leakyrelu", in_features=3*32*32):
    return MLP(in_features, 
                10,
                [ 512, 512],
                [act_func, act_func, act_func],
                use_batch_norm=True,
                use_dropout=True,
                use_softmax=True,
                device=device)