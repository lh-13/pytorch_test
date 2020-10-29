'''
@Author: lh-13
@Date: 2020-10-10 16:27:03
@LastEditTime: 2020-10-12 16:01:44
@LastEditors: Please set LastEditors
@Description: pytorch nn 模块 试验
@FilePath: \pytorch_test\pytorch_nn_test.py
'''

import torch  
import torch.nn as nn  
import torchvision.transforms as transforms
import os  
import numpy as np  
from PIL import Image 
import random 


def set_seed(seed):
    torch.manual_seed(seed)     #cpu 为cpu设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)   #gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True      #cudnn 
    np.random.seed(seed)      #numpy
    random.seed(seed)         #random and transforms   



set_seed(1)

#测试nn.Conv2d
img = Image.open('./cat.jpg')
img.show()
print(img.size)

#convert to tensor
img_tensor = transforms.ToTensor()(img)
#convert to 4 dim 
img_tensor.unsqueeze_(dim=0)     #C*H*W -> B*C*H*W 
img_conv_tensor = nn.Conv2d(3, 3, 3)
nn.init.xavier_normal_(img_conv_tensor.weight.data)       #it's very important 

img_conv_tensor = img_conv_tensor(img_tensor)



#convert to PILImage
img_conv_tensor = img_conv_tensor.squeeze(dim=0)
img_conv = transforms.ToPILImage()(img_conv_tensor)
img_conv.show()

#测试nn.MaxPool2d 和nn.AvgPool2d 
maxPool_image_tensor = nn.MaxPool2d((2, 2))(img_conv_tensor)
#convert to PILImage 
img_pool_tensor = maxPool_image_tensor.squeeze(dim=0)
img_pool = transforms.ToPILImage()(img_pool_tensor)
img_pool.show()



