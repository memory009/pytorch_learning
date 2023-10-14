# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
#引入乳腺癌的数据集
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
# X.shape输出为（569,30）
# n_feature = 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# random_state参数主要是为了保证每次都分割一样的训练集和测试机，大小可以是任意一个整数，在调参缓解，只要保证其值一致即可
# print(X_train.shape)   #输出（455,30）
# print(X_test.shape)    #输出（114,30）确实是按照80%，20%分
# print(y_train.shape)   # 输出（455,）
# print(y_test.shape)    # 输出（114,）


# scale
# standarscalar 标准定标器，使之具有0均值和0方差，做回归的时候很需要这个操作  需要提前import
sc = StandardScaler()

# 根据对之前部分trainData进行fit的整体指标，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，
# 从而保证train、test处理方式相同。
# Fit(): Method calculates the parameters μ and σ and saves them as internal objects. 求得训练集X的均值啊，方差等训练集X固有的属性。可以理解为一个训练过程
# Transform(): Method using these calculated parameters apply the transformation to a particular dataset. 在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）
# Fit_transform(): joins the fit() and transform() method for transformation of dataset. fit_transform是fit和transform的组合，既包括了训练又包含了转换。
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 将训练数据和测试数据转换为张量，只有转换为tensor形式才能在pytorch中训练
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# 此处步骤类似于reshape()，就是将y_train转换为xx行，1列的形式，与X_train,x_test保持一致
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)
# print(y_test)

# 1)model
# f = wx + b, sigmod at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
# print(model)
# LogisticRegression((linear): Linear(in_features=30, out_features=1, bias=True))

# 2) loss and optimizer
learing_rate = 0.01

# 二元交叉熵损失（笔记在notability）
criterion = nn.BCELoss()
# SGD 随机梯度下降
optimizer = torch.optim.SGD(model.parameters(),lr = learing_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss =criterion(y_predicted, y_train)

    # backward pass
    loss.backward()
    # print(loss)
    # 直接输出loss会包含他的类型 grad_fn=<BinaryCrossEntropyBackward>，所以要输出loss的话需要用loss.item()


    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# 在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
# tensor的requires_grad的属性默认为False,
# 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
# with torch.no_grad的作用,在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
with torch.no_grad(): 
    y_predicted = model(X_test)
    # .round() 四舍五入计算,保证y_predicted_cls输出不是0就是1
    y_predicted_cls = y_predicted.round()
    # print(y_predicted_cls)
    # print(y_test)
    # print(y_test.shape[0])
    # print(y_test.shape[0]),输出114，意思是输出y_test.shape中的第一维，第一个【】中包含114个二维数组，即【】
    # torch.eq()对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；
    # 若不同，返回False,进行sum()之后求出具体有多少个值是相同的，然后除以总的数量得出acc
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    # print(acc)
    # tensor(0.8860)
    print(f'accuracy = {acc.item():.4f}')

# 测试下git的功能