# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update gradients
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# regression 回归
# 0) Prepare data
# datasets.make_regression()（产生一个随机回归问题）,n_samples = 样本数， n_features = 特征数， noise = 应用于输出的高斯噪声的标准偏差。
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# X_numpy : (n_samples, n_features)  即随机生成100行1列的数组
# y_numpy : (n_samples)  即随机生成100个数
# print(X_numpy.shape)

# astype(将numpy转换为tensor样式，以进行后续操作)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# y是100个数，所以需要转换为100行1列的二维数组，使之和x_numpy一样才有可比性，
# 因为X_numpy是(100,1)，y_numpy一开始是(100,)，所以对y进行view操作，使之变成(100,1)
y = y.view(y.shape[0],1)


n_samples, n_features = X.shape
# print(X.shape)

# 1) Model
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
# MSE均方误差
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# print(model.parameters)
# <bound method Module.parameters of Linear(in_features=1, out_features=1, bias=True)>

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    #update
    optimizer.step()

    #zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
# detach()分离函数， .numpy()作用是将tensor格式转成numpy格式，让matplotlib可以直接用
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()



