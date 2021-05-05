#!/bin/python

# Custom classes
from models.mlp import MLP, DefaultCifar10MLP
from utils.dataloader import * # Get the transforms and loaders for cifar-10
from utils.train import Trainer # Default custom training class
from utils.accuracy_plot import AccuracyPlot
from models.resnet import *
from models.cnn import CNN

#TODO rerun to report high accuracy!

# Use GPU!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:  " + str(device))
#device=None # Disable GPU

##############################################
##############  Dataset Loading ##############
##############################################

# Get the transformers
# Use different train/test data augmentations
transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# See utils/dataloader.py for data augmentations
# transform_train = GetTrainTransform()
transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Get Cifar 10 Datasets
save='./data/Cifar10'
trainset = LoadCifar10DatasetTrain(save, transform_train)
testset = LoadCifar10DatasetTest(save, transform_test)

# Get Cifar 10 Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

epochs = 25

# copy form Assignment 6 in case
#################################
# From the last assignment
#################################
def GetDefaultCNN():
    cnn = CNN( in_features=(32,32,3),
                out_features=10,
                conv_filters=[32,32,64,64],
                conv_kernel_size=[3,3,3,3],
                conv_strides=[1,1,1,1],
                conv_pad=[0,0,0,0],
                max_pool_kernels=[None, (2,2), None, (2,2)],
                max_pool_strides=[None,2,None,2],
                use_dropout=False,
                use_batch_norm=False,
                actv_func=[None, "relu", None, "relu"],
                device=device
        )
    # Create MLP
    # Calculate the input shape
    s = cnn.GetCurShape()
    in_features = s[0]*s[1]*s[2]

    mlp = MLP(in_features,
                10,
                [],
                [],
                use_batch_norm=False,
                use_dropout=False,
                use_softmax=False,
                device=device)

    # mlp = DefaultCifar10MLP(device=device, in_features=in_features)

    cnn.AddMLP(mlp)
    return cnn

model = GetDefaultCNN()
summary(model, (3,32,32))

#################################
# Create the Plotter
#################################
# Custom dynamic plotting class #
plotter = AccuracyPlot()


#################################
# Train the model, plot accuracy
#################################
# Optimizer
opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.95))

# Loss function
cost = nn.CrossEntropyLoss()

# Create a trainer
trainer = Trainer(model, opt, cost, name="Default CNN", device=device)

# Add test accuracy plotting
plotter.new_line("Default CNN")
trainer.SetEpochCallback(plotter.EpochCallback)

# Run training
trainer.Train(trainloader, epochs, testloader=testloader)


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

print('done')
