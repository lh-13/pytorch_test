'''
@Author: your name
@Date: 2020-10-17 10:32:03
@,@LastEditTime: ,: 2020-11-12 16:40:59
@,@LastEditors: ,: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pytorch_test\PyTorch_test.py
'''
"""
this is a test for pytorch 
author:lh-13
date:2020.0203
"""

import torch

# a = torch.Tensor(2, 2)
# print(a)
# print(a.shape)
# print(a.numel())

# b = torch.DoubleTensor(2, 2)
# print(b)

# c = torch.Tensor([[1, 2], [3, 4]])
# print(c)
# print(c.shape)
# print(c[1])
# print(c[0, 1])

# print('---------------------')
# print(c>0)   #打印出>0的位置
# print(c[c>0])  #选择符合条件的元素并返回
# print(torch.nonzero(c))   #选择非0元素的坐标并返回

# #torch.where(condition, x, y), 满足condition的位置输出x,否则输出y
# print(torch.where(c>1, torch.full_like(c, 2), c))

# d = torch.randn(2, 2)
# print(d)



#tensor 的排序与取极值
#sort()
flag = 0
if flag == 1:
    a = torch.randn(2,2)
    print(a)

    print(a.sort(0, True)[0])   #按照第0维即按行进行排序， 每一列进行比较，True代表降序， False代表升序
    print('--------------------------------------')
    print(a.sort(1, True)[0])
    print(a.sort(0, True)[1])  #???   应该是排序后矩阵之前的索引


#torch.sort 对向量进行排序
flag = 0  
if flag == 1: 
    a = torch.tensor([50, 20, 30, 10])
    print(a)

    b = torch.sort(a)
    print(b)
    print('第一维度是排序后的向量：{}'.format(b[0]))
    print('第二维度是排序后对应排序前向量的索引：{}'.format(b[1]))


flag = 1 #  测试下 tensor ==        
if flag == 1:
    tensor_1 = torch.randn((4))
    print(tensor_1)
    min_value = min(tensor_1.abs())
    print(min_value)
    percent = (min_value == tensor_1.abs()).nonzero().item() / len(tensor_1)
    print( percent)




