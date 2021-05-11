
import torch
from torch.utils.data import DataLoader, ConcatDataset
# from sklearn.model_selection import KFold
# from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
from pylab import *
import os
from utils.dynamic_plot import DynamicUpdate
from torch.utils.tensorboard import SummaryWriter
import subprocess

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from datetime import datetime
from tensorboard import program



class Trainer():
    def __init__(self, net, opt, cost, name="default", lr=0.0005, start_epoch=0, use_lr_schedule =False , device=None, decode_out=None):
        self.net = net
        self.opt = opt
        self.cost = cost
        self.device = device
        self.epoch = 0
        self.start_epoch = 0
        self.name = name
        self.decode_out = decode_out
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

    # Run a single batch with inputs and labels on the net
    # Calculate the loss gradient and take a optimizer step
    def TrainBatch(self, inputs, epoch):
        # Zero the parameter gradients
        self.opt.zero_grad()

        # Forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.cost(outputs, inputs.view(-1, 28 * 28))#, reduction='sum')
        # loss = F.binary_cross_entropy(outputs, inputs.view(-1, 28 * 28))

        loss.backward()
        # loss.sum().backward()
        self.opt.step()

        if epoch % 10 == 0 and (self.decode_out is not None):
            self.net.save_decod_img(outputs.cpu().data, epoch, file=self.decode_out)

        return loss, outputs

    def TrainEpoch(self, trainloader, epoch, testloader=None):
        self.net.train() # Enable Dropout
        epoch_loss = 0.0
        for data in trainloader:
            # Get the inputs; data is a list of [inputs, labels]
            if self.device:
                images, _ = data[0].to(self.device), data[1].to(self.device)
            else:
                images, _ = data

            loss, outputs = self.TrainBatch(images, epoch)
            # print(loss)

            epoch_loss += loss.item()
            # epoch_loss /=  len(trainloader)

        return epoch_loss


    # Train loop over epochs. Optinal use testloader to return test accuracy after each epoch
    def Train(self, trainloader, epochs, testloader=None):

        # Tensorboard init
        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
        directory = os.path.dirname(os.getcwd() + '/runs/tensorboard-%s/' % (date_time))
        print(f'Run the following code in bash to view progress: \n tensorboard --logdir=\'{directory}\'')

        if not os.path.exists(directory):
            os.makedirs(directory)

        writer = SummaryWriter(log_dir=directory)

        # Enable Dropout
        self.net.train()

        # Record loss/accuracies
        loss = torch.zeros(epochs)
        self.epoch = 0

        # Train for each epoch
        for epoch in range(self.start_epoch, self.start_epoch +epochs):
            # If testloader is used, loss will be the accuracy
            loss[epoch] = self.TrainEpoch(trainloader, epoch=epoch+1, testloader=testloader)

            # Save the data
            self.epoch += 1

            # Print Epoch Loss
            print("Epoch %d Learning rate %.6f Loss: %.5f" %
            (self.epoch, self.opt.param_groups[0]['lr'], loss[epoch]))
            info = { 'Loss/train': loss[epoch]}
            writer.add_scalar('Loss/train', info['Loss/train'], global_step=epoch)

            # # Saving last batch
            # if epoch == 20:
            #     pic = self.net.show_image(output.cpu().data)
            #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

        return loss

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

    def save_checkpoint(self, state):
        directory = os.path.dirname("./save/%s-checkpoints/" % (self.name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(state, "%s/model_epoch_%s.pt" % (directory, self.epoch))
        # torch.save(state, "./save/checkpoints/model_epoch_%s.pt" % (self.epoch))



