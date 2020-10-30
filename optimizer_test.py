'''
@Author: lh-13
@Date: 2020-10-17 16:18:09
@LastEditTime: 2020-10-18 11:22:00
@LastEditors: Please set LastEditors
@Description: 优化器的一些例子与优化器的策略选择
@FilePath: \pytorch_test\optimizer_test.py
'''

import torch  
import torch.nn as nn  
import torch.optim as optim
import os  
import matplotlib.pyplot as plt
import numpy as np  


torch.manual_seed(1)  

def func(x_t):
    '''
    y = (2x)^2=4*x^2  dy/dx = 8x 
    '''
    return torch.pow(2*x_t,2)


x = torch.tensor([2.], requires_grad=True)
print(x)
#--------------------------------------------------------plot data
# x_t = torch.linspace(-3, 3, 100)
# y = func(x_t)
# plt.plot(x_t.numpy(), y.numpy(), label="y=4*x^2")
# plt.grid()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

#--------------------------------------------------------gradient descent
# iter_rec, loss_rec, x_rec = list(), list(), list() 
# lr = 0.01 
# max_iteration = 20  
# for i in range(max_iteration):
#     y = func(x)
#     y.backward() 
#     print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))
    
#     x_rec.append(x.item())
#     x.data.sub_(lr*x.grad)    #x -= x.grad #0.5, 0.2, 0.1, 0.125
#     x.grad.zero_()

#     iter_rec.append(i)
#     loss_rec.append(y)

# plt.subplot(121).plot(iter_rec, loss_rec, "-ro")
# plt.xlabel("Iteration")
# plt.ylabel("Loss Value")

# x_t = torch.linspace(-3, 3, 100)
# y = func(x_t)
# plt.subplot(122).plot(x_t.detach().numpy(), y.detach().numpy(), label="y = 4*x^2")
# plt.grid()
# y_rec = [func(torch.tensor(i)).item() for i in x_rec]
# plt.subplot(122).plot(iter_rec, y_rec, "-ro")
# plt.legend()
# plt.show()


#----------------------------------------------------multi learning rate 
# iteration = 100
# num_lr = 10
# lr_min, lr_max = 0.01, 0.2  # .5 .3 .2

# lr_list = np.linspace(lr_min, lr_max, num=num_lr).tolist()
# loss_rec = [[] for l in range(len(lr_list))]
# iter_rec = list()

# for i, lr in enumerate(lr_list):
#     x = torch.tensor([2.], requires_grad=True)
#     for iter in range(iteration):

#         y = func(x)
#         y.backward()
#         x.data.sub_(lr * x.grad)  # x.data -= x.grad
#         x.grad.zero_()

#         loss_rec[i].append(y.item())

# for i, loss_r in enumerate(loss_rec):
#     plt.plot(range(len(loss_r)), loss_r, label="LR: {}".format(lr_list[i]))
# plt.legend()
# plt.xlabel('Iterations')
# plt.ylabel('Loss value')
# plt.show()

#----------------------------------------------------优化器策略选择（下降策略）

LR = 0.1
iteration = 10
max_epoch = 200

#---------------------------------------------------fake data and optimizer 
weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1)) 
print(weights)
print(target)

optimizer = torch.optim.SGD([weights],lr=LR, momentum=0.9)

#----------------------------------------------------StepLR  等间隔调整学习率
'''
lr_scheduler_StepLR(optimizer, step_size, gamma=0.1, last_epoch)
step_size 表示调整间隔
gamma 表示调整系数，调整方式就是lr=lr*gramma.这里的gamma一般为0.1-0.5 
用的时候只需要指定调整间隔，比如50，那么就是50个epoch调整一次学习率，调整方式就是lr=lr*gamma 
'''

# secheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50)    #设置学习率下降策略

# lr_list, epoch_list = list(),list() 
# for epoch in range(max_epoch):
#     lr_list.append(secheduler_lr.get_lr())
#     epoch_list.append(epoch)

#     for i in range(iteration):
#         loss = torch.pow((weights-target),2)
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

#     secheduler_lr.step()   #更新学习策略

# plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
# plt.xlabel("Epoch")
# plt.ylabel("learning rate")
# plt.legend()
# plt.show()


#----------------------------------------------------Multi Step LR  按给定间隔调整学习率
'''
lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
milestones 表示设定调整时刻数，可以是一个list，表示调整的间隔,比如：[50, 125, 150],50个epoch、125个epoch、150个epoch调整一次学习率
gamma 调整方式：lr=lr*gamma 
'''

# milestones = [50, 125, 160]
# scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
# lr_list, epoch_list = list(), list()  
# for epoch in range(max_epoch):
#     lr_list.append(scheduler_lr.get_lr())
#     epoch_list.append(epoch)

#     for i in range(iteration):
#         loss = torch.pow((weights-target), 2)
#         loss.backward()

#         optimizer.step() 
#         optimizer.zero_grad()

#     scheduler_lr.step()  

# plt.plot(epoch_list, lr_list, label='Multi Step LR Scheduler\nmilestones:{}'.format(milestones))
# plt.xlabel("epoch")
# plt.ylabel("Learning rate")
# plt.legend()
# plt.show()

#----------------------------------------------------Exponential LR 按指数衰减调整学习率
'''
lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
gamma表示指数的底了。调整方式：lr=lr*gamma^epoch  
'''

# gamma = 0.95 
# scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# lr_list, epoch_list = list(), list()   

# for epoch in range(max_epoch):
#     lr_list.append(scheduler_lr.get_lr())
#     epoch_list.append(epoch)

#     for i in range(iteration):
#         loss = torch.pow((weights-target), 2)
#         loss.backward()

#         optimizer.step() 
#         optimizer.zero_grad()  

#     scheduler_lr.step()  

# plt.plot(epoch_list, lr_list, label='Exponential LR Scheduler\n')
# plt.xlabel("Epoch")
# plt.ylabel("Learning rate")
# plt.legend()  
# plt.show() 

#----------------------------------------------------CosineAnnealingLR  余弦周期调整学习率
'''
lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

'''

# t_max = 50  
# scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
# lr_list, epoch_list = list(), list()  

# for epoch in range(max_epoch):
#     lr_list.append(scheduler_lr.get_lr())
#     epoch_list.append(epoch)

#     for i in range(iteration):
#         loss = torch.pow((weights-target), 2)
#         loss.backward()

#         optimizer.step()  
#         optimizer.zero_grad()

#     scheduler_lr.step()   

# plt.plot(epoch_list, lr_list, label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
# plt.xlabel("Epoch")
# plt.ylabel("Learning rate")
# plt.legend()
# plt.show()


#----------------------------------------------------ReduceLRonPlateau  监控指标，当指标不再变化则调整。可以监控loss或者准确率，当不在变化的时候，我们再去调整
'''
lr_scheduler.ReduceLRonPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,min_lr=0, eps=1e-08)



















