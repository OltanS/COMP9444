#!/usr/bin/env python3
"""

@UNSW COMP9444 2021/T3 Assignment 2
14/11/2021

@authors 
Shakeel Anver z5172568
Oltan Sevinc z5230739 

student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

Acknowledgements:

RESNET implementation adapted from
https://niko-gamulin.medium.com/resnet-implementation-with-pytorch-from-scratch-23cf3047cb93
modified (and simplified) to fit the assignment application domain
Approach chosen after research into most efficient methods for image classification


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),        
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur((5, 5), sigma=(0.1, 0.3))]), p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.Pad(padding=5), transforms.Resize((80,80))]), p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(degrees=(0, 45), translate=(0,0.2), scale=(0.5, 0.75))]), p=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],std=[0.5, 0.5,0.5])
        ])
        
        return tf
    elif mode == 'test':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],std=[0.5, 0.5,0.5])
        ])
        return tf


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, input):
        identity = input
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.conv2(input)
        input = self.bn2(input)

        # resize input so it can be added to the output of the layer
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        input += identity
        input = self.relu(input)
        return input

class ResNet(nn.Module):
    def __init__(self, output_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 3 because of 3 channel image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # create the blocks
        self.bl1 = self.make_blocks(3, 64, stride=1)
        self.bl2 = self.make_blocks(4, 128, stride=2)
        self.bl3 = self.make_blocks(6, 256, stride=2)
        self.bl4 = self.make_blocks(3, 512, stride=2)

        self.fc = nn.Linear(512, output_classes)
        self.drop = nn.Dropout(p=0.25)
    
    def forward(self, input):
        # initial processing before blocks
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.maxpool(input)
        input = self.relu(input)

        # run it through the blocks
        input = self.bl1(input)
        input = self.bl2(input)
        input = self.bl3(input)
        input = self.bl4(input)

        # final processing
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        input = avg_pool(input)
        # linearise for linear layer
        input = input.view(input.shape[0], -1)
        input = self.drop(input)
        input = self.fc(input)
        return input  

    def make_blocks(self, num_blocks, intermediate_channels, stride):
        '''
        Creates the blocks for the resnet structure
        '''
        layers = []
        # downsample so the addition operations work between differently sized input and output
        # for the resnet skipping to function
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels))
        layer = Layer(self.in_channels, intermediate_channels, identity_downsample, stride)
        layers.append(layer)
        # update in_channels to set up next layers correctly
        self.in_channels = intermediate_channels # 256
        for i in range(num_blocks - 1):
            layer = Layer(self.in_channels, intermediate_channels)
            layers.append(layer) # 256 -> 64, 64*4 (256) again
        # unpack layers list and make a block of all the sequential layers
        return nn.Sequential(*layers)



net = ResNet(8)

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(),lr=0.005, betas=(0.9,0.999), weight_decay=0.0001)

loss_func = nn.CrossEntropyLoss()

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.

def weights_init(m):
    return

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.2)


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 32
epochs = 500