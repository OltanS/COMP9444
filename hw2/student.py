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
            # transforms.RandomAffine(degrees=30, translate=(0.1,0.2), scale=(0.7, 1)),
            transforms.RandomHorizontalFlip(),        
            # transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomPerspective(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],std=[0.5, 0.5,0.5])
        ])
        
        return tf
    elif mode == 'test':
        tf = transforms.Compose([
            # transforms.RandomAffine(degrees=30, translate=(0.1,0.2), scale=(0.7, 1)),
            transforms.RandomHorizontalFlip(),        
            # transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomPerspective(),
            transforms.RandomVerticalFlip(),
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


class BottleNeckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(BottleNeckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1,stride=stride,padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, input):
        identity = input
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        # print(f"input shape after first conv layer: {input.shape}")
        input = self.conv2(input)
        input = self.bn2(input)
        # print(f"input shape after second conv layer: {input.shape}")
        input = self.conv3(input)
        input = self.bn3(input)

        # resize input so it can be added to the output of the layer
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # print(f"identity shape: {identity.shape}")
        # print(f"input shape: {input.shape}")
        input += identity
        input = self.relu(input)
        return input


class ResNet(nn.Module):
    def __init__(self, output_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 3 because of 3 channel image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # create the blocks
        self.bl1 = self.make_blocks(3, 64, stride=1)
        self.bl2 = self.make_blocks(4, 128, stride=2)
        self.bl3 = self.make_blocks(6, 256, stride=2)
        self.bl4 = self.make_blocks(3, 512, stride=2)
        # self.bl3 = self.make_bottleneck_blocks(6, 256, stride=1)
        # self.bl4 = self.make_bottleneck_blocks(3, 512, stride=1)

        self.fc = nn.Linear(512, output_classes)
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.7),
        #     nn.Linear(512, output_classes)
        # )
    
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
        input = self.fc(input)
        return input  


    def make_bottleneck_blocks(self, num_blocks, intermediate_channels, stride):
        '''
        Creates the bottleneck blocks for the resnet structure to  make things more efficient
        when dealing with larger features
        '''
        layers = []
        # downsample so the addition operations work between differently sized input and output
        # for the resnet skipping to function
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels))
        layer = BottleNeckLayer(self.in_channels, intermediate_channels, identity_downsample, stride)
        layers.append(layer)
        # update in_channels to set up next layers correctly
        self.in_channels = intermediate_channels # 256
        for i in range(num_blocks - 1):
            layer = BottleNeckLayer(self.in_channels, intermediate_channels)
            layers.append(layer) # 256 -> 64, 64*4 (256) again
        # unpack layers list and make a block of all the sequential layers
        return nn.Sequential(*layers)


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
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9,0.999))
# optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9, weight_decay=0.007, nesterov=True)

loss_func = nn.CrossEntropyLoss()

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.

# from https://androidkt.com/initialize-weight-bias-pytorch/
def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight.data, 1)
    #     nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.Linear):
    #     nn.init.kaiming_uniform_(m.weight.data)
    #     nn.init.constant_(m.bias.data, 0)
    return

scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# lambda1 = lambda epoch: epoch/10
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 32
epochs = 200

'''
THINGS TO TRY:

CONV2D -> MAXPOOL -> RELU instead of CONV2D -> RELU -> MAXPOOL -> Done

Spatial Dropout -> with Dropout2d https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html -> Idk how to use

Data augmentation -> torch.transforms -> Done

Learning Rate Scheduler -> ExponentialLR, MultistepLR etc https://pytorch.org/docs/stable/optim.html -> Done -> Disabled now

SGD + Momentum with weight initialization https://pytorch.org/docs/stable/generated/torch.optim.SGD.html -> Done

Initialize with  https://androidkt.com/initialize-weight-bias-pytorch/ -> Done

'''