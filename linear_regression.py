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

# 0) Prepare data
# X_numpy = [[2.0, 6.0, 5.0, 1.0], [7.0, 9.0, 3.0, 2.0]]
X_numpy = [[2.0, 7.0], [6.0, 9.0], [5.0, 3.0], [1.0, 2.0]]
y_numpy = [52.8, 96.7, 21.2, 6.0]


# astype(将numpy转换为float样式，以进行后续操作)
X = torch.from_numpy(np.array(X_numpy)).float()
y = torch.from_numpy(np.array(y_numpy)).float()

# X = X.transpose(0,1)
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape
# print(n_samples)
# print(n_features)
n_y = y.shape[1]
# print(X.shape)

# 1) Model
input_size = n_features
output_size = n_y
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

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
# detach()分离函数， .numpy()作用是将tensor格式转成numpy格式，让matplotlib可以直接用
predicted = model(X).detach().numpy()
# print(predicted)




fig1 = plt.figure()
col_1 = [[row[0]] for row in X_numpy]
plt.plot(col_1, y_numpy, 'ro')
plt.plot(col_1, predicted, 'r')


fig2 = plt.figure()
col_2 = [[row[1]] for row in X_numpy]
plt.plot(col_2, y_numpy, 'bo')
plt.plot(col_2, predicted, 'b')
plt.show()