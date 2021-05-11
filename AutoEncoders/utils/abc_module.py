import abc
import torch.nn as nn
from utils.utils import GetActivation
import torch
import torch.nn.functional as F

class Modulelist_(abc.ABC, nn.Module):
    """List Module"""

    def __init__(self,
                 in_features,
                 layers,
                 actv_func,
                 use_dropout,
                 use_batch_norm,
                 use_softmax,
                 init_method=None,
                 decoder=False,
                 image_size=None
                ):
        """Initialization"""

        super().__init__()
        self.in_features = in_features
        self.layers = layers
        self.num_layers = len(self.layers)
        self.actv_func = actv_func
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.use_softmax = use_softmax
        self.use_init = False
        self.image_size = image_size
        if init_method is not None:
            self.init_method = init_method
            # self.init_method = self.GetDefaultInitialization()
            # self.init_method = {key: init_method.get(key, self.init_method[key]) for key in self.init_method}
            self.use_init = True

        self.decoder = decoder
        self.module_list = nn.ModuleList()

    def build_(self):

        dim = self.in_features**2 if self.decoder is False else self.in_features

        for i in range(self.num_layers):
            # Create a MLP Encoder/Decoder
            self.module_list.append(nn.Linear(dim, self.layers[i]))
            # Update the current dimension
            dim = self.layers[i]
            if self.use_batch_norm:
                self.module_list.append( nn.BatchNorm1d(dim, affine=True) )
            # Add the Activation function
            self.module_list.append( GetActivation(name=self.actv_func[i]) )
            if self.use_dropout:
                self.module_list.append( nn.Dropout(p=0.15) )

        if self.decoder:
            # Fully connect to output dimensions
            if dim != self.image_size:
                self.module_list.append(nn.Linear(dim, self.image_size**2))

        if self.use_init:
            self.init_layer()

        #print(self.module_list)
        return self.module_list

    def init_layer(self):
        # Initialization #TODO add in diff class
        print(self.module_list)

        for m in self.module_list:
            if isinstance(m, nn.Linear):
                self.GetInitialization(weights= m.weight, name=self.init_method)

    def forward(self,x):
        # Flatten the 2d image into 1d only for Encoder
        # Also convert into float for FC layer
        if self.decoder is False:
            # x = torch.flatten(x.float(), start_dim=1)
            x = x.view(-1, 28 * 28)

        # Apply each layer in the Encoder Module list
        for i in range( len(self.module_list) ):
            x = self.module_list[i](x)

        if self.decoder and self.use_softmax :
            if self.use_softmax:
                # x = F.softmax(x, dim=1)
                x = torch.sigmoid(x)
        return x

    def GetInitialization(self, weights, name="normal"):
        initialization_methods = ['normal', 'kaiming_uniform', 'xavier_normal']
        assert name in initialization_methods, "Choose a valid init method: (normal, kaiming_uniform or xavier_normal) "

        if name == 'normal':
            return nn.init.normal_(weights, mean=0.0, std=1.0) #TODO add gin config
        elif name == "kaiming_uniform":
            return nn.init.kaiming_uniform_(weights, a=0, mode='fan_in')
        elif name == "xavier_normal":
            return nn.init.xavier_normal_(weights, gain=1.0)
        elif name == 'xavier_uniform':
            return nn.init.xavier_uniform(weights)


    def GetDefaultInitialization(self):
        init_method ={'Sigmoid': 'normal',
                       'ReLU': 'kaiming_uniform',
                       'Tanh': 'xavier_normal'}
        return init_method


# nn.init.kaiming_normal_(m.weight)
# nn.init.normal_(m.weight, mean=0.0, std=0.0)
# nn.init.normal_(m.weight, mean=0.0, std=0.0)