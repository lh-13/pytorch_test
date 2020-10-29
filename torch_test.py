'''
@Author: your name
@Date: 2020-09-16 10:25:34
@LastEditTime: 2020-09-16 10:48:02
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \PyTorch_YOLOv4-master\torch_test.py
'''
import torch  

idx = torch.tensor([2, 3], dtype=torch.long)
print(idx)
print(idx.shape)

a = torch.tensor((2, 3))
print(a)
print(a.shape)

b = torch.ones((2, 3))
print(b)
print(b.shape)