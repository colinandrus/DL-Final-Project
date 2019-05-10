#!/usr/bin/env python
# coding: utf-8

# In[20]:


import argparse
import os
import random
import shutil
import time
import warnings
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


from resnet_segmentation import *


# In[21]:


torch.cuda.get_device_name(0)
device = 'cuda'


# In[22]:


data_dir = "/scratch/cra354/ssl_data_96/supervised"
# using smaller, dataset for dev
#data_dir = "/scratch/cra354/smaller_set/"
model_save_path = "/home/cra354/resnet18_classifier_denoise2.pth"


# In[23]:


input_shape = 320
batch_size = 16
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 320
use_parallel = True
use_gpu = True
epochs = 100

data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        #transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        #transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


# In[24]:


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).type(torch.FloatTensor).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


# In[25]:


model_conv = nn.Sequential(GaussianNoise(), ResNet18())
model_conv.load_state_dict(torch.load('resnet_reconstruction_2.pth'))


# In[26]:


resnet_nofc = nn.Sequential(*list(model_conv[1].children())[:8])


# In[27]:


model = nn.Sequential(resnet_nofc, nn.AdaptiveAvgPool2d((1)), Reshape(batch_size, 2048) , nn.Linear(2048,1000))


# In[28]:


# train  params all layers
#for param in model.parameters():
#    param.requires_grad = False
# require grad for last layer
#ct = 0
#for child in model.children():
#    ct += 1
#    if ct < 3:
#        for param in child.parameters():
#            param.requires_grad = False


# In[29]:


criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, 
                            model_conv.parameters())), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# In[30]:


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu=True, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                #augementation using mixup
                if phase == 'train' and mixup:
                    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                #if phase == 'val':
                    #print("preds: {}".format(preds))
                    #print("labels: {}".format(labels))
                    #print(epoch_acc)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                #running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / torch.tensor(dataset_sizes[phase]).float().cuda()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), model_save_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[31]:


model = model.cuda()


# In[ ]:


model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu=True,
                     num_epochs=100)


# In[ ]:




# In[ ]:




