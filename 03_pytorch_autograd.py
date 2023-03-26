import torch
#torch.rand均匀分布，torch.randn是标准正态分布
x = torch.randn(3,requires_grad=True)
# print(x)


#如果不想输出梯度，可以使用下列方法，但其实默认就是不输出梯度信息的
#1: x.requires_grad_(False)
#2: x.detach()
#3. with torch.no_grad():

###################################################################

#example:
import torch
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    #获取梯度信息
    model_output.backward()
    print(weights.grad)
    #每次运算后需要清空梯度
    weights.grad.zero_()

