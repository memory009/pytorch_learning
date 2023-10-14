import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # here the first column is the class label, the rest are the features
        self.x = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        self.n_samples = xy.shape[0] # 获取行信息，有多少行就有多少个samples

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = WineDataset()
# num_workers=2 使用两个子处理器,win笔记本电脑端需要注释掉才能运行
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# iter() 函数用来生成迭代器。
datatiter = iter(dataloader)
# print(datatiter)
# <torch.utils.data.dataloader._SingleProcessDataLoaderIter object at 0x000001988C548AC0>
data = datatiter.next()
features, labels = data
# print(features)
# 因为batch_size=4 所以一次读四行
# tensor([[1.1460e+01, 3.7400e+00, 1.8200e+00, 1.9500e+01, 1.0700e+02, 3.1800e+00,
#          2.5800e+00, 2.4000e-01, 3.5800e+00, 2.9000e+00, 7.5000e-01, 2.8100e+00,        
#          5.6200e+02],
#         [1.2530e+01, 5.5100e+00, 2.6400e+00, 2.5000e+01, 9.6000e+01, 1.7900e+00,        
#          6.0000e-01, 6.3000e-01, 1.1000e+00, 5.0000e+00, 8.2000e-01, 1.6900e+00,
#          5.1500e+02],
#         [1.4100e+01, 2.0200e+00, 2.4000e+00, 1.8800e+01, 1.0300e+02, 2.7500e+00,
#          2.9200e+00, 3.2000e-01, 2.3800e+00, 6.2000e+00, 1.0700e+00, 2.7500e+00,
#          1.0600e+03],
#         [1.2520e+01, 2.4300e+00, 2.1700e+00, 2.1000e+01, 8.8000e+01, 2.5500e+00,
#          2.2700e+00, 2.6000e-01, 1.2200e+00, 2.0000e+00, 9.0000e-01, 2.7800e+00,
#          3.2500e+02]])

# print(labels)
# tensor([[1.],
#         [2.],
#         [1.],
#         [2.]])

# training loop
num_epochs = 2 
total_samples = len(dataset)
n_interation = math.ceil(total_samples/4)
# print(total_samples, n_interation)
# 178 45

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward , update
        if (i+1) % 5 == 1:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_interation}, inputs {inputs.shape}')


