

import torch
from torch.utils.data import DataLoader, ConcatDataset
# from sklearn.model_selection import KFold
# from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
from pylab import *
import os
from utils.dynamic_plot import DynamicUpdate

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR



class Trainer():
    def __init__(self, net, opt, cost, name="default", lr=0.0005, start_epoch=0, use_lr_schedule =False , device=None):
        self.net = net
        self.opt = opt
        self.cost = cost

        self.device = device

        self.epoch = 0
        self.start_epoch = 0


        self.name = name
        
        self.BatchCallback = None
        self.EpochCallback = None
        self.EpochCallbackList = []
        self.BatchCallbackList = []
        self.RegularizationCallbackList = []
        self.live_plot = None
        self.lr = lr
        self.use_lr_schedule = use_lr_schedule
        if self.use_lr_schedule:
            self.scheduler = ReduceLROnPlateau( self.opt, 'max', factor=0.1, patience=10, threshold=0.00001, verbose=True)
            # self.scheduler = StepLR(self.opt, step_size=15, gamma=0.1)

    def LivePlotSetup(self, epochs):
        self.live_plot = DynamicUpdate()
        self.live_plot.set_max(epochs)
        self.live_plot.on_launch()

    def UpdatePlot(self, epoch, accuracies):
        self.live_plot.on_running(range(epoch), accuracies)

    # Customizable batch callback,
    # Callback is handed the net, optimizer, and cost function.
    # ex Declaration:  
    #       def EpochCallback(net=None, opt=None, cost=None):
    #           ...
    #       trainder.SetEpochCallback( EpochCallback )
    # ex. Can be used to print out the gradian norm of each layer
    def SetEpochCallback(self, callback):
        self.EpochCallbackList.append(callback)

    # ex Declaration:  
    #       def BatchCallback(net=None, opt=None, cost=None, loss=None, inputs=None, labels=None):
    #           ...
    #       trainder.SetBatchCallback( BatchCallback )
    def SetBatchCallback(self, callback):
        self.BatchCallbackList.append(callback)

    # Add regularization.  It is called with net and added to the loss
    def AddRegularization(self, callback):
        self.RegularizationCallbackList.append(callback)


    # Configure when using a different starting epoch than 0
    # Maybe used when loading and continuing training on a model
    def SetStartEpoch(self, epoch):
        self.start_epoch = epoch
        self.epoch = self.start_epoch

    # Run a single batch with inputs and labels on the net
    # Calculate the loss gradient and take a optimizer step
    def TrainBatch(self, inputs, labels):
        # Zero the parameter gradients
        self.opt.zero_grad()

        # Forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.cost(outputs, labels)

        # Add Regularization
        for callback in self.RegularizationCallbackList:
            loss += callback(self.net)

        loss.backward()
        self.opt.step()

        # Run Calback before we go
        for callback in self.BatchCallbackList:
            callback(net=self.net, opt=self.opt, cost=self.cost, 
                                loss=loss, inputs=inputs, labels=labels)

        return loss

    # Train an Epoch
    # if the testloader is defined, return the test accuracy
    #  instead of the train error
    def TrainEpoch(self, trainloader, testloader=None):
        self.net.train() # Enable Dropout
        epoch_loss = 0.0
        for data in trainloader:
            # Get the inputs; data is a list of [inputs, labels]
            if self.device:
                images, labels = data[0].to(self.device), data[1].to(self.device)
            else:
                images, labels = data

            loss = self.TrainBatch(images, labels)

            epoch_loss += loss.item()
                

        if testloader:
            epoch_loss = self.Test(testloader)
        else:
            epoch_loss /= len(trainloader)


        return epoch_loss


    # Train loop over epochs. Optinal use testloader to return test accuracy after each epoch
    def Train(self, trainloader, epochs, testloader=None):
        # Enable Dropout
        self.net.train()

        # Record loss/accuracies
        loss = torch.zeros(epochs)

        self.epoch = 0

        # Train for each epoch
        for epoch in range(self.start_epoch, self.start_epoch+epochs):
            # If testloader is used, loss will be the accuracy
            loss[epoch] = self.TrainEpoch(trainloader, testloader=testloader)

            #learning rate scheduler
            if self.use_lr_schedule:
                self.scheduler.step(loss[epoch])
                # self.scheduler.step()

            # Run the epoch callback
            for callback in self.EpochCallbackList:
                callback(epoch=epoch, accuracies=np.array(loss), net=self.net, opt=self.opt, cost=self.cost)


            # Update the plot
            if self.live_plot:
                self.UpdatePlot(epoch+1, loss[:epoch+1])

            # Save the data 
            self.epoch += 1

            # Save the model
            # self.net.SaveModel('./save/%s/cifar_net_epoch_%s' % (self.name, self.epoch))

            #Different saving method
            self.save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.net.state_dict(),
                'optimizer': self.opt.state_dict(),
            })

            # Print Epoch Loss
            print("Epoch %d Learning rate %.6f %s: %.3f" % (self.epoch, self.opt.param_groups[0]['lr'], "Accuracy" if testloader else "Loss", loss[epoch]) )

        return loss

    def save_checkpoint(self, state):
        directory = os.path.dirname("./save/%s-checkpoints/"%(self.name))
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(state, "%s/model_epoch_%s.pt" %(directory, self.epoch))
        # torch.save(state, "./save/checkpoints/model_epoch_%s.pt" % (self.epoch))


    # Test over testloader loop
    def Test(self, testloader, ret="accuracy"):
        # Disable Dropout
        self.net.eval() 

        # Track correct and total
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in testloader:
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    # Test activation maps
    def Test_filter(self, testloader, ret="accuracy"):
        # Disable Dropout
        self.net.eval()

        # Track correct and total
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in testloader:
                if self.device:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                else:
                    images, labels = data

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return outputs



class CrossValidTrainer( Trainer ):
    def __init__(self, model_factory, opt_factory, cost_function, name="default", device=None):
        self.ModelFactory = model_factory
        self.OptFactory = opt_factory

        # Default data loader used to convert dataset into dataloader in cross validation
        def DefaultDataLoader(data):
            return torch.utils.data.DataLoader(data, 
                        batch_size=64, 
                        num_workers=4,
                        shuffle=True)
        self.DataLoader = DefaultDataLoader

        net = self.ModelFactory(device)
        opt = self.OptFactory(net.parameters())

        super().__init__(net, opt, cost_function, name=name, device=device)

        # Create a callback to run at each split
        self.SplitCallbackList = []

    def SetSplitCallback(self, callback):
        self.SplitCallbackList.append(callback)
    

    def CrossValidTrain(self, trainset, epochs, kfold=5):
        # Create each split $
        # Get the size of each split
        fold_size = len(trainset) // kfold
        splits = torch.utils.data.random_split(trainset, (fold_size,) * kfold)

        # Record accurcy on the validation set of each split for each epoch
        accuracies = torch.zeros((kfold, epochs))

        for split in range(kfold):
            # Get the splits
            train_dataset = ConcatDataset([*splits[:split], *splits[split+1:]])

            # And the data loaders
            trainloader = self.DataLoader(train_dataset)
            validloader = self.DataLoader(splits[split])

            # Create our model and optimizer
            self.net = self.ModelFactory(self.device)
            self.opt = self.OptFactory(self.net.parameters())

            # Run training over train/valid set
            accuracies[split,:] = self.Train(trainloader, epochs, testloader=validloader)

            # Run the callback
            for callback in self.SplitCallbackList:
                callback()

            # Create a new plot line
            if self.live_plot:
                self.live_plot.new_line(split)

        return np.array(accuracies.mean(dim=0))

    def EarlyStopping(self, trainset, epochs, kfold=5, UseCrossValidation=True):
        # Run cross valid training
        accuracies = self.CrossValidTrain(trainset, epochs, kfold=kfold)

        # Get the index of the model with the best accuracy
        best_epoch = np.argmax(accuracies)
        # Re-create our best model by loading parameters from the best epoch
        # Each epoch is saved by:
        #       self.net.SaveModel('./save/%s/cifar_net_epoch_%s' % (self.name, self.epoch))
        best_net = self.ModelFactory(self.device)
        param_file = './save/%s/cifar_net_epoch_%s' % (self.name, best_epoch)
        best_net.RestoreModel(param_file)

        # Return the epoch and net
        return best_epoch, best_net
