import numpy as np 

# Compute every step manually(手动计算每一步)

# Linear regression(线性回归)
# f = w * x 
# here : f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model output
def forward(x):
    return w * x

# loss = MSE(均方误差)
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# y_pred = w * x
# J = MSE_loss = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    # np.dot() 点乘
    # np.mean() 取平均值
    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01

#迭代次数
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass(前向传播)
    y_pred = forward(X)

    #loss（上面定义了loss函数）
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)  # #dl/dw，相当于损失函数关于权重 w 的梯度，和04中w.grad一样

    #update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch{epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

