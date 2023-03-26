'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):

    def __init__(self,transform=None):
        # __init__方法,把参数transform绑上去
        # Initialize data, download, etc.
        # read with numpy or pandas
        # skiprows 跳过的行数为1, delimiter将','作为分割
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here
        self.x = xy[:, 1:]  # size [n_samples, n_features]
        self.y = xy[:, [0]] # size [n_samples, 1]
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sampele):
        inputs, targets = sampele
        # return在这里的语句可以使self.x，self.y，即sample具有tensor的格式
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


# dataset = WineDataset(transform=None)
# 如果这里是None的话，输出<class 'numpy.ndarray'> <class 'numpy.ndarray'>
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
# <class 'torch.Tensor'> <class 'torch.Tensor'>

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))