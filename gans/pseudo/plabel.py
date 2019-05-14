import argparse
import os
import random
import shutil
import time
import warnings
import sys
import random
#from tqdm import tqdm

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
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from rand_mask import RandomMasking
from resnet_segmentation import *


torch.cuda.get_device_name(0)
device = 'cuda'

#train_dir    = "/home/thekingh/final_projects/gans/dl_data/supervised/train"
train_dir    = "/scratch/ks4883/dl_data/supervised/train"
#ul_dir       = "/home/thekingh/final_projects/gans/dl_data/unsupervised/"
train_dir    = "/scratch/ks4883/dl_data/supervised/train"
ul_dir       = "/scratch/ks4883/dl_data/unsupervised/"
IMG_SIZE = 64
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
classes, classes_to_idx = train_data._find_classes(train_dir)
#classes, classes_to_idx = datasets.folder.find_classes(train_dir)
idx_to_class = {}
for class_name in classes_to_idx:
    index = classes_to_idx[class_name]
    idx_to_class[index] = class_name
print("num classes: ", len(classes))

unlabel_data   = datasets.ImageFolder(ul_dir, transform=transform)
unlabel_loader = torch.utils.data.DataLoader(unlabel_data, batch_size=batch_size, shuffle=True)
print("loaded unlabel data, batch size = {0}, num_batches = {1}".format(batch_size, len(unlabel_loader)))

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

#print(torch.load("weights.pth").keys())
model_conv = nn.Sequential(RandomMasking(), ResNet18())
#model_conv.load_state_dict(torch.load("weights.pth"))
resnet_nofc = nn.Sequential(*list(model_conv[1].children())[:8])
model = nn.Sequential(resnet_nofc, nn.AdaptiveAvgPool2d((1)), Reshape(batch_size, 2048) , nn.Linear(2048,1000))
model.load_state_dict(torch.load("weights.pth"))
model.cuda()
print("created model")

#p_label_root = "/home/thekingh/final_projects/gans/dl_data/plabel/"
p_label_root = "/scratch/ks4883/dl_data/plabel/"
if not os.path.exists(p_label_root):
    os.mkdir(p_label_root)

def save_plabel(img, num_label):

    class_label = idx_to_class[num_label.item()]

    root = p_label_root
    if not os.path.exists(root+class_label):
        os.mkdir(root+class_label)

    identifier = random.randint(1, 1000)
    save_image(img.data, root+class_label+"/{0}_{1}.jpg".format(class_label, identifier))

#unlabel_iter = iter(unlabel_loader)
#for num_batch in range(len(unlabel_loader)):
num_iteration = 0

cutoff_2 = 0
cutoff_4 = 0
cutoff_6 = 0
cutoff_8 = 0
cutoff_10 = 0
cutoff_12 = 0

THRESHOLD = 6.
batch_num = 0
for imgs, _ in unlabel_loader:    
     
    imgs = Variable(imgs.cuda())
    output = model(imgs)
    vals, pred_labels = torch.max(output.data, 1)

    for img, pred_label, val in zip(imgs, pred_labels, vals):
        prob = val.item()

        if val > THRESHOLD:
            save_plabel(img, pred_label)

    batch_num += 1
    print("finished batch, ", batch_num)
