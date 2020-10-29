'''
pytorch tensor test
'''

import torch 
import numpy as np 

#创建tensor 5*3 空矩阵
# a = torch.empty(5,3)
# print(a)

# #创建随机tensor
# #torch.rand()  返回在区间[0, 1]上均匀分布的随机数填充的张量
# b = torch.rand((5, 3))
# print(b)

# # c = torch.tensor([1, 2, 3])
# # print(c)

# #torch.randn 返回一个从均值为0、方差为1的正太分布(也称标准正太分布)
# c = torch.randn((5, 3))
# print(c)


#创建一个全0的tensor
# d = torch.zeros((5, 3), dtype=torch.int16)
# print(d)
# print(d.type())
# print(d.size())

#张量的运算
#加法
#x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# print(x)
# print(x[:, 1])
# print(x[..., 1:3])
# print(x[..., 1:])

# print(x)
# print(y)
# print(x+y)
# print(torch.add(x,y))
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
# torch.add(x, y, alpha=10, out=result)    #out = x + y*alpha  
# print(result)
# print(torch.add(x, 20))

# print(x)
# print(y)
# y.add_(x)
# print(y)

#resize 
# y = x.view(-1, 5)
# print(x)
# print(y)

#tensor与numpy转换
a = torch.ones(5, 3)
print(a)
b = a.numpy()
print(b)
a=torch.add(a,3)
print(a)
print(b)

# c = np.ones((5, 3))
# print(c)
# d = torch.from_numpy(c)
# print(d)
# np.add(c, 1, out=c)
# print(c)
# print(d)




