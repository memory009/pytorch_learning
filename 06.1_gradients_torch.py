# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update gradients

import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# X_test是一个测试值，意思是如果输入5，会输出什么结果
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
# print(n_samples, n_features)  #此处结果为4 1 

input_size = n_features
output_size = n_features

# model 调用torch.nn做了一个线性回归
# model = nn.Linear(input_size, output_size)
# print(model)
#输出Linear(in_features=1, out_features=1, bias=True)
# in_features：输入张量的size，out_features:输出张量的size， bias：偏置项

# print(model(X_test))
#输出tensor([-1.7815], grad_fn=<AddBackward0>)

##########################################################
#如果要使用自己的数据集
class LinearRegression(nn.Module):

    def __init__(self,input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

##########################################################

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01

#迭代次数
n_iters = 100

# MSE均方误差
loss = nn.MSELoss()
# SGD 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass(前向传播)
    y_pred = model(X)

    #loss（上面定义了loss函数）
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    #每迭代多少次，输出一个结果
    if epoch % 10 == 0:
        [w, b] = model.parameters()
#         print(w)
#         输出Parameter containing:
#         tensor([[1.7053]], requires_grad=True)
#         所以下面写成w[0][0].item()为了可以输出w的具体值
        # print(b)
        print(f'epoch{epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

