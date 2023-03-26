import torch

# Compute every step manually(手动计算每一步)

# Linear regression(线性回归)
# f = w * x 
# here : f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)

# model output
def forward(x):
    return w * x

# loss = MSE(均方误差)
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01

#迭代次数
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass(前向传播)
    y_pred = forward(X)

    #loss（上面定义了loss函数）
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch{epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

