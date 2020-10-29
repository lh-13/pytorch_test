'''
@Author: lh-13
@Date: 2020-10-15 11:14:13
@LastEditTime: 2020-10-16 11:01:03
@LastEditors: Please set LastEditors
@Description: pytorch 中一些loss函数的使用
@FilePath: \pytorch_test\lossfunction_test.py
'''

import torch  
import torch.nn as nn  

#CrossEntropyLoss 交叉熵损失函数 
'''
nn.CrossEntropyLoss(weight=None, size_average=None,ignore_index=-100, reduce=None, reduction=mean)
weight:各类别的loss设置权值
ignore_index:忽略某个类别
reduction:计算模式，可为none/sum/mean,none表示逐个元素计算，这样有多少个样本就会返回多少个loss。sum表示所有元素的loss求和，返回标量，
mean所有元素的loss求加权平均,返回标量
'''


input = torch.tensor([[1,2], [1, 3], [1, 3], [1,2]], dtype=torch.float)    #这里就是模型预测的输出
target = torch.tensor([0, 1, 1, 0], dtype=torch.long)                  #这里的类型必须是long，两个类0和1,个数应与上面对应上面为4个，这里也应为4个

print(input)
print(target)

weight = torch.tensor([1, 2], dtype=torch.float)   #这里的weights是个张量，并且按照类别设置权值(几个classes就为几个)

#三种模式的损失函数
loss_f_none = nn.CrossEntropyLoss(weight=weight, reduction='none')
loss_f_sum = nn.CrossEntropyLoss(weight=weight, reduction='sum')
loss_f_mean = nn.CrossEntropyLoss(weight=weight, reduction='mean')



#forward
loss_none = loss_f_none(input, target)
loss_sum = loss_f_sum(input, target)
loss_mean = loss_f_mean(input, target)

#view 
print("Cross Entropy Loss:\n ", loss_none, loss_sum, loss_mean)


#doc example:
# loss = nn.CrossEntropyLoss()
# input = torch.randn((3, 5), requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)   #产生0-5  之间的随机数
# print(input)
# print(target)

# output = loss(input, target)
# output.backward()
# print("output:\n", output)


