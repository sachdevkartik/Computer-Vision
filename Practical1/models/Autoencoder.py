#!/bin/python
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.utils import GetActivation
from utils.abc_module import Modulelist_

import os
import matplotlib.pyplot as plt
import numpy as np
#TODO init_method: {'Sigmoid': "normal"}

@gin.configurable()
class AutoEncoder(nn.Module):
    def __init__(self
            , in_features
            , encoder_layers
            , decoder_layers
            , encoder_actv_func
            , decoder_actv_func
            , image_size
            , use_dropout=False
            , use_batch_norm=False
            , use_softmax=True
            , device="cpu"
            , init_method= None
            ):
        super(AutoEncoder, self).__init__()

        assert encoder_layers[-1] == decoder_layers[0], "Encoder Last Layer & Decoder First layer shoudl be of same size"

        self.encoder_module = EncoderModule(in_features=in_features, encoder_layers=encoder_layers, encoder_actv_func=encoder_actv_func,
                                            use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                            use_softmax=use_softmax, init_method=init_method)

        self.decoder_module = DecoderModule(in_features=encoder_layers[-1], decoder_layers=decoder_layers, decoder_actv_func=decoder_actv_func,
                                            use_dropout=False, use_batch_norm=False, use_softmax=use_softmax,
                                            init_method=init_method, image_size=image_size)

        self.module_list = nn.ModuleList()
        self.build_()

        # Send to gpu

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def build_(self):
        encoder_list = self.encoder_module.build_()
        self.module_list.append(encoder_list)

        decoder_list = self.decoder_module.build_()
        self.module_list.append(decoder_list)
        print(self.module_list)

    def forward(self, x):
        x = self.encoder_module.forward(x)
        x = self.decoder_module.forward(x)
        return x

    def show_image(self, img):
        plt.imshow(img.numpy().reshape(28, 28), cmap='gray')
        plt.show()

    def save_decod_img(self, img, epoch, file):
        img = img.view(img.size(0), 1, 28, 28)
        # img = img * np.array(0.1307,) + np.array(0.3081,)
        directory = os.path.dirname(file)

        if not os.path.exists(directory):
            os.makedirs(directory)

        save_image(img, f'{directory}/Image_epoch_{epoch}.png')

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

class EncoderModule(Modulelist_):
    def __init__(self,
                 in_features,
                 encoder_layers,
                 encoder_actv_func,
                 use_dropout=False,
                 use_batch_norm=False,
                 use_softmax=False,
                 init_method=None #TODO assertion error of options available
                 ):
        """Initializing the master"""

        super().__init__(
                in_features=in_features,
                layers= encoder_layers,
                actv_func = encoder_actv_func,
                use_dropout = use_dropout,
                use_batch_norm = use_batch_norm,
                use_softmax = use_softmax,
                init_method = init_method
                )

    # def forward(self,x):
    #     super().forward(x)

class DecoderModule(Modulelist_):
    def __init__(self,
                 in_features,
                 decoder_layers,
                 decoder_actv_func,
                 use_dropout=False,
                 use_batch_norm=False,
                 use_softmax=False,
                 init_method=None,
                 image_size=None
                 ):
        """Initializing the master"""

        super().__init__(
                in_features=in_features,
                layers= decoder_layers,
                actv_func = decoder_actv_func,
                use_dropout = use_dropout,
                use_batch_norm = use_batch_norm,
                use_softmax = use_softmax,
                init_method = init_method,
                decoder=True,
                image_size=image_size
                )

    # def forward(self,x):
    #     super().forward(x)



def DefaultCifar10MLP(device="cpu", act_func="leakyrelu", in_features=3*32*32):
    return MLP(in_features, 
                10,
                [ 512, 512],
                [act_func, act_func, act_func],
                use_batch_norm=True,
                use_dropout=True,
                use_softmax=True,
                device=device)