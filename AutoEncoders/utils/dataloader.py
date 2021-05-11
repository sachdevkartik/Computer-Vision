
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import random

# Get the data
def LoadMNIST(save, transforms_, batch_size=32, ret_sets=False):

    trainset = torchvision.datasets.MNIST(root=save, train=True,
                                        download=True, transform=transforms_) #transforms.ToTensor()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root=save, train=False,
                                       download=True, transform=transforms_)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if ret_sets:
        return trainloader, testloader, classes, trainset, testset
    return trainloader, testloader, classes


def imshow(im):

    image = im.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # unnormalize
    plt.imshow(image)
    plt.show()

def imretrun(im):

    image = im.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # unnormalize
    return image

# Fasion Mninst:
#  28 x 28 grayscale images
#  10 clothing types
def LoadFashionMNISTTrain(save, transform):
    trainset = torchvision.datasets.FashionMNIST(root=save, train = True, download = True, transform = transform)
    return trainset

def LoadFashionMNISTTest(save, transform):
    testset = torchvision.datasets.FashionMNIST(root=save, train = False, download = True, transform = transform)
    return testset


# Get the data
def LoadCifar10(save, transforms, batch_size=32, ret_sets=False):

    trainset = torchvision.datasets.CIFAR10(root=save, train=True,
                                        download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=save, train=False,
                                       download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if ret_sets:
        return trainloader, testloader, classes, trainset, testset
    return trainloader, testloader, classes

def LoadCifar10DatasetTrain(save, transform=None):
    trainset = torchvision.datasets.CIFAR10(root=save, train=True,
                                        download=True, transform=transform)
    return trainset

def LoadCifar10DatasetTest(save, transform):
    return torchvision.datasets.CIFAR10(root=save, train=False,
                                       download=True, transform=transform)

#################################
# Transformations on trainset(part b)
#################################

def GetTrainTransform():
    # Create a ton of different types of transforms
    transform_crop = transforms.Compose([
            transforms.CenterCrop((30,30)), # Crop 30 x 32
            transforms.Pad((1,1))  # Re-pad to 32 x 32
        ])

    # transform_color = transforms.ColorJitter()

    transform_single = transforms.Compose([
        # transform_crop,
        # transforms.GaussianBlur((5,5), sigma=(0.1, 2.0)),
        # transforms.RandomHorizontalFlip(p=1.0), #First training
        # transforms.RandomVerticalFlip(p=0.5), #Second training
        # transforms.RandomAffine(30), #First training
        transforms.RandomRotation(20), #Third training
        # transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.1), #Second training
        transforms.RandomCrop(32, (2,2), pad_if_needed=False, padding_mode='constant'), #4,4 to 2,2
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform_rotate = transforms.RandomAffine(30)
    # transform_translate = transforms.RandomAffine(0, translate=(0.1,0.1))
    # transform_scale = transforms.Compose([
    #         transforms.RandomAffine(0, scale=(1.0,1.2)),
    #         transforms.CenterCrop((32,32))
    #     ])
    #
    # transform_shear = transforms.RandomAffine(0, shear=40)
    # transform_flip = transforms.RandomHorizontalFlip(p=1.0)
    # # transform_blur = transforms.GaussianBlur((5,5), sigma=(0.1, 2.0))
    #
    # # Add them together to a list
    # transform_list = [transform_crop, transform_color, transform_gray,
    #                     transform_rotate, transform_translate,
    #                     transform_scale, transform_shear,
    #                     transform_flip]
    #
    # transform_prob = 1.0 / len(transform_list)
    #
    # # Create another list where we choose to use each of the
    # #   listed transfroms with probability 'transform_prob'.
    # random_transform = []
    # for transform in transform_list:
    #     random_transform.append( transforms.RandomApply([transform], transform_prob) )
    #
    # # Compose the final transform
    # train_transform = transforms.Compose([
    #                     transforms.RandomApply(random_transform, 0.9999),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                     ])
    return transform_single


#################################
# Refactor dataset class (part b)
# Tried to make dataset class but had problem with dataloader
# Hence, this class is only for checking the transformation done on the train set
#################################

class CustomLoader(Dataset):
    def __init__(self, save, train_transform):
        self.train_transform = train_transform
        self.dict_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.path = save
        self.img, self.label = self._get_batch_label()

    def _get_batch_label(self):
        img = None
        label = []
        for _, batch in enumerate(self.dict_batch):
            unpickled_file = self._unpickle(os.path.join(self.path, batch))
            unpickled_img = unpickled_file[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            if img is None:
                img = unpickled_img
            else:
                img = np.vstack((unpickled_img, img))
            label += unpickled_file[b'labels']
        return img, label

    def _unpickle(self, file):

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    #For a list of transformation
        # for transform in self.transform_list:
        #     dataset = load_func(save, transform)
        #     self.datasets.append(dataset)
        #     self.length += len(dataset)

    def __len__(self):
        # return self.length
        return (len(self.label))

    def __getitem__(self, idx):
        # if len(self.datasets) == 1:
        #     return self.datasets[0][idx]
        # dataset_idx = np.random.randint(len(self.label))
        img = self.img[idx]
        img = self.train_transform(img)
        label = self.label[idx]

        return img, label
        # return self.datasets[dataset_idx][idx]


'''
transform_normal = transforms.Compose([
        transforms.CenterCrop(1000),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_color = transforms.Compose([
        transforms.CenterCrop(1000),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_flip = transforms.Compose([
        transforms.RandomResizedCrop(1000),
        # transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_rotate = transforms.Compose([
        transforms.CenterCrop(1000),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_blur = transforms.Compose([
        transforms.CenterCrop(1000),
        # transforms.GaussianBlur((7,7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
'''
# transform_rotate = transforms.Compose([
#         transforms.RandomResizedCrop(1000),
#         transforms.RandomAffine(20),
#         transforms.RandomVerticalFlip(p=0.999),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

# transform_crop = transforms.Compose([
#         transforms.CenterCrop((30,30)), # Crop 30 x 32
#         transforms.Pad((1,1))  # Re-pad to 32 x 32
#     ])
# transform_color = transforms.ColorJitter()
# transform_gray = transforms.Grayscale(num_output_channels=3)
# transform_rotate = transforms.RandomAffine(30)
# transform_translate = transforms.RandomAffine(0, translate=(0.1,0.1))
# transform_scale = transforms.Compose([
#         transforms.RandomAffine(0, scale=(1.0,1.2)), 
#         transforms.CenterCrop((32,32))  
#     ])
# transform_shear = transforms.RandomAffine(0, shear=40)
# transform_flip = transforms.RandomHorizontalFlip(p=1.0)
# # transform_blur = transforms.GaussianBlur((5,5), sigma=(0.1, 2.0))

# transform_list = [transform_crop, transform_color, transform_gray,
#                     transform_rotate, transform_translate, 
#                     transform_scale, transform_shear, 
#                     transform_flip]

# transform_prob = 1.0 / len(transform_list)

# # add random apply to each transform
# random_transform = []
# for transform in transform_list:
#     random_transform.append( transforms.RandomApply([transform], transform_prob) )
# train_transform = transforms.Compose([
#                     transforms.RandomApply(random_transform, 0.9999),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                     ])


# save='./data/Cifar10'

# train_transform = transforms.Compose([
#                 transforms.RandomApply([
#                         # transforms.Compose([
#                         #     transforms.CenterCrop((30,30)), # Crop 30 x 32
#                         #     # transforms.Pad((2,2))  # Re-pad to 32 x 32
#                         #     ]),
#                         # torch.nn.Sequential(
#                         #     transforms.CenterCrop(900), # Crop 30 x 32
#                         #     transforms.Pad((1,1,1,1))  # Re-pad to 32 x 32
#                         #     ),
#                         # transforms.GaussianBlur((7,7), sigma=(0.1, 2.0)),
#                         # transforms.RandomAffine(0, shear=40),  # Shear
#                         transforms.ColorJitter()
#                     ], 0.9999),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                 ])


# image_datasets = torchvision.datasets.CIFAR10(root=save, train=True, 
#                                             download=True, transform=train_transform)
# train_dataloader =  torch.utils.data.DataLoader(image_datasets, batch_size=32,
#                                              shuffle=True, num_workers=4)
# dataiter = iter(train_dataloader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


'''

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(1000),
        # transforms.RandomAffine(20),
        transforms.RandomVerticalFlip(p=0.999),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}


image_datasets = {x: torchvision.datasets.CIFAR10(root=save, train=x=='train', 
                                            download=True, transform=data_transforms[x])
                for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

classes = image_datasets['train'].classes




# Example Usage:
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''



# # Load the Cifar10 data
# transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainloader, testloader, classes= LoadCifar10(save, transform, ret_sets=False)

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))