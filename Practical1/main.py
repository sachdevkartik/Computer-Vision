#!/bin/python

# Custom classes
from utils.dataloader import * # Get the transforms and loaders for cifar-10
from utils.train import Trainer # Default custom training class
from models.Autoencoder import *
from utils.accuracy_plot import AccuracyPlot

# Standard imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import argparse
from torch.nn import functional as F
import gin

def main(config):
    # Use GPU!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:  " + str(device))

    # Simple transform
    transform = transforms.Compose([transforms.ToTensor()]) #transforms.Normalize((0.1307,), (0.3081,)

    # Get MNIST Datasets
    save = config.data_loc #'./data/MNIST' #'/home/kartik/git/ADLG/Assignment2/data/MNIST'
    trainloader, testloader,_ = LoadMNIST(save, transforms_=transform,batch_size=config.batch_size)

    epochs = config.epochs
    print(config.epochs)

    model = AutoEncoder()

    # opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.95))
    opt = optim.SGD(model.parameters(), lr=config.lr)

    # Loss function
    # cost = config.loss
    cost = eval(config.loss + "()") # nn.MSELoss()
    loss_name = eval(config.loss + '.__name__')
    # cost = nn.BCELoss() #reduce=False

    decode_out = config.save_images + loss_name + '/' #'./save/Decoder_Images/'

    # Create a trainer
    trainer = Trainer(model, opt, cost, name="Default Autoencoder",
                      device=device, decode_out=decode_out)

    # Run training
    trainer.Train(trainloader, epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', help='Specify number of epochs', default=20, type=int)
    parser.add_argument(
        '--loss', default= 'nn.MSELoss', choices=['nn.MSELoss', 'nn.BCELoss', 'nn.CrossEntropyLoss'],
        help='Specify loss function without brackets e.g. nn.MSELoss)')
    parser.add_argument(
        '--lr', help='Specify learning rate', default=0.005, type=float)
    parser.add_argument(
        '--save-images', help='Location to store decoder images',
        default=f'./save/Decoder_Images/')
    parser.add_argument(
        '--batch-size', help='Determine batch size', default=32, type=int)
    parser.add_argument(
        '--data-loc', help='Specify location of stored decoder images ',
                        default='./data/MNIST')
    parser.add_argument(
        '--momentum', default=0.9, type=float, help='Specify momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument(
        '--config', help='The file used to store the AE gin config settings', 
        default='./config/auto_config.gin')

    args, _ = parser.parse_known_args()

    gin.parse_config_file(args.config)

    main(args)

'''
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#################################
# Create the assignment Resnet (part a)
#################################
def GetDefaultResNet():
    resnet = ResNet(in_features= [32, 32, 3],
                    num_class=10,
                    feature_channel_list = [32, 64, 128],
                    batch_norm= True,
                    )

    # Create MLP
    # Calculate the input shape
    s = resnet.GetCurShape()
    in_features = s[0]*s[1]*s[2]

    # mlp = MLP(in_features,
    #             10,
    #             [],
    #             [],
    #             use_batch_norm=False,
    #             use_dropout=False,
    #             use_softmax=False,
    #             device=device)

    mlp = DefaultCifar10MLP(device=device, in_features=in_features)

    resnet.AddMLP(mlp)
    return resnet

model = GetDefaultResNet()
summary(model, (3, 32,32))

#################################
# Train the model, plot accuracy
#################################
# Optimizer
opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.95))

# Loss function
cost = nn.CrossEntropyLoss()

# Create a trainer
trainer = Trainer(model, opt, cost, name="Default ResNet", device=device)

# Add test accuracy plotting
plotter.new_line("Default ResNet")
trainer.SetEpochCallback(plotter.EpochCallback)

# Run training

trainer.Train(trainloader, epochs, testloader=testloader)


plotter.show()

'''