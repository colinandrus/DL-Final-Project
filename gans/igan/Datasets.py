import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np

IMG_SIZE = 64

transform = transforms.Compose(
    [
       transforms.Resize(IMG_SIZE),
       transforms.CenterCrop(IMG_SIZE),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def MnistLabel(class_num):
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def MnistUnlabel():
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    return raw_dataset
def MnistTest():
    return datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))

def DL_Label(path, num_per_class):
    print("fetching supervised data from {0}/{1}".format(path, 'supervised'))
    sup_train_data = datasets.ImageFolder('{0}/{1}/train'.format(path, 'supervised'), transform=transform)

    data    = []
    labels  = []

    class_total    = [0] * 1000
    positive_total = 0
    total          = 0

    perm = np.random.permutation(sup_train_data.__len__())
    for i in range(sup_train_data.__len__()):
        data_pt, label = sup_train_data.__getitem__(perm[i])
        if class_total[label] < num_per_class:
            data.append(data_pt.numpy())
            labels.append(label)
            class_total[label] += 1
            total += 1
            if total >= 1000 * num_per_class:
                break

    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))
            

def DL_Unlabel(path):
    unsup_data = datasets.ImageFolder('{0}/{1}'.format(path, 'unsupervised'), transform=transform)
    return unsup_data

def DL_Test(path):
    sup_val_data = datasets.ImageFolder('{0}/{1}/val'.format(path, 'supervised'), transform=transform)
    return sup_val_data

if __name__ == '__main__':
    print( dir(MnistTest()))
