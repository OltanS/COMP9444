#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

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
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        
    def forward(self, input):
        pass

class DenseNet(nn.Module):

    def __init__(self, num_hid):
        super().__init__()
        self.fc1 = nn.Linear(3*80*80,num_hid)
        self.fc2 = nn.Linear(num_hid + (3*80*80), num_hid)
        self.fc3 = nn.Linear(num_hid + num_hid + (3*80*80), 8)

        
    def forward(self, input):
        # print(f"input shape: {input.shape}")
        input = input.view(200,-1)
        # print(f"adjusted input shape: {input.shape}")
        self.hid1 = torch.tanh(self.fc1(input))
        self.hid2 = torch.tanh(self.fc2(torch.cat((input,self.hid1),dim=1)))
        self.output = F.log_softmax(self.fc3(torch.cat((input,self.hid1,self.hid2),dim=1)),dim=1)
        # print(f"Output Shape: {self.output.shape}")
        return self.output

# net = Network()
net = DenseNet(50)
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(),lr=0.005)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 200
epochs = 200
