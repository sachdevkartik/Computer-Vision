#!/bin/python

# Custom classes
from models.mlp import MLP, DefaultCifar10MLP
from utils.dataloader import * # Get the transforms and loaders for cifar-10
from utils.train import Trainer # Default custom training class
from utils.accuracy_plot import AccuracyPlot
from models.resnet import *
from models.cnn import CNN


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
transform_train = GetTrainTransform()
trainset = LoadCifar10DatasetTrain(save, transform_train)
# trainset = LoadCifar10DatasetTrain(save, transform_train)

# Get Cifar 10 Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=4)

# num_workers = 6,
# prefetch_factor = 4, persistent_workers = True)
testset = LoadCifar10DatasetTest(save, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=4) # TODO check num_workers , prefetch_factor = 4, persistent_workers = True

epochs = 60
#################################
# Create the Plotter
#################################
# Custom dynamic plotting class #
plotter = AccuracyPlot()

#################################
# Create the assignment Resnet (part b)
#################################
def GetDefaultResNet():
    resnet = ResNet(in_features= [32, 32, 3],
                    num_class=10,
                    feature_channel_list = [32, 64, 128],
                    batch_norm= True,
                    num_stacks=2
                    )

    # Create MLP
    # Calculate the input shape
    s = resnet.GetCurShape()
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

    resnet.AddMLP(mlp)
    return resnet

#################################
# Load the trained model
#################################
model = GetDefaultResNet()
summary(model, (3, 32,32))

model_loaded = GetDefaultResNet()
# PATH = "./save/Modified-ResNet2-checkpoints/model_epoch_50.pt" #Model Acc = 83.4%

PATH = "./save/ResNet2_accuracy-85_checkpoints/model_epoch_50.pt" #Model Acc = 85%

checkpoint = torch.load(PATH)
model_loaded.load_state_dict(checkpoint['state_dict'])
loaded_epoch= checkpoint['epoch']
summary(model_loaded, (3, 32,32))

#################################
# Train the model, plot accuracy 
#################################
# Optimizer
lr = 0.0005

# opt.load_state_dict(checkpoint['optimizer'])
opt = optim.Adam(model_loaded.parameters(), lr=lr, betas=(0.9, 0.999))
opt.param_groups[0]['lr'] = 0.005 #0.005

# opt = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# Loss function
cost = nn.CrossEntropyLoss()

# Create a trainer
trainer = Trainer(model_loaded, opt, cost, name="ResNet2-accuracy_above_85",lr=lr , use_lr_schedule=True, device=device)

# Add test accuracy plotting
plotter.new_line("Default ResNet")
trainer.SetEpochCallback(plotter.EpochCallback)

# Run training
epochs = 50
trainer.Train(trainloader, epochs, testloader=testloader)
# trainer.Train(trainloader, epochs) # check train error


plotter.show()

print('done')
