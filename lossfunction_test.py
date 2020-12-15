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

flag = 0 
if flag == 1:
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


'''
torch.nn.KLDivLoss()   KL散度损失（用于连续分布的距离度量，并且对离散采用的连续输出空间分布进行回归通常很有用）
公式：参考https://blog.csdn.net/qq_36533552/article/details/104034759
公式理解：
p(x)是真实分布，q(x)是拟合分布；实际计算时，通常p(x)作为target，只是概率分布；而xn则是把输出做了LogSoftmax计算；即把概率分布映射到log空间；
所以 K-L散度值实际是看log(p(x))-log(q(x))的差值，差值越小，说明拟合越相近
'''

import torch   
import torch.nn as nn
import numpy as np  

#--------------------------------------------------------------------KLDiv loss 
loss_f = nn.KLDivLoss(size_average=False, reduce=False)      
loss_f_mean = nn.KLDivLoss(size_average=True, reduce=True)    #reduce=True,返回所有元素loss的和； size_average=True,返回所有元素的平均值，优先级比reduce高

#生成网络输出以及目标输出 
output = torch.from_numpy(np.array([[0.1132, 0.5477, 0.3390]])).float()  
output.requires_grad = True  
print(output)
#target = torch.from_numpy(np.array([[0.8541, 0.0511, 0.0947]])).float()   
target = torch.tensor([0.8541, 0.0511, 0.0947])
print(target)

loss_1 = loss_f(output, target)
loss_f_mean = loss_f_mean(output, target)
print('loss:', loss_1)
print('\n')
print('loss_mean:', loss_f_mean)



