import torch

x = torch.rand(2,2)
y = torch.rand(2,2)
# print(x)
# print(y)

#矩阵相加，两种写法
z = x + y
z = torch.add(x,y)
# print(z)

#此时对y进行操作，相当于修改了y的值
y.add_(x)
# print(y)

#矩阵相减两种方法
z = x - y
z = torch.sub(x,y)
# print(z)

#矩阵相乘两种方法
z = x * y
z = torch.mul(x,y)
# print(z)

#矩阵相除两种方法
z = x / y
z = torch.div(x,y)
# print(z)

#####################################################################
#矩阵的切片获取
x = torch.rand(5, 3)
# print(x)
#遍历所有行，每一行都取第0个元素
# print(x[:,0])
#遍历所有行，每一行都取第0个元素
# print(x[0,:])
#读第一行，第一列的元素
# print(x[1, 1])
#只输出第一行，第一列的元素的值
# print(x[1, 1].item()
#####################################################################

#在gpu运算
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device) #x一步将矩阵放进gpu，y用两步
    y = torch.ones(5)
    y = y.to(device)
    z = x + y   #因为x&y都在gpu上了，所以这样运算会快很多
    #numpy不能在gpu中直接使用，需要转回CPU再用
    z = z.to("cpu")