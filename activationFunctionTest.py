import torch 
import torch.nn.functional as F 
from torch.autograd import Variable

import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)  #构造一段连续的数据
x = Variable(x)   #转换成张量
x_np = x.data.numpy()   #转化为numpy格式，plt中形式需要numpy



# #sigmoid 函数
# y_sigmoid = F.sigmoid(x).data.numpy()
# plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
# plt.ylim(-0.2, 1.2)
# plt.legend(loc='best') #图例名称自动选择最佳展示位置
# plt.show()  


#tanh 函数
# y_tanh = F.tanh(x).data.numpy()
# plt.plot(x_np, y_tanh, c='red', label='tanh')
# plt.ylim(-1.2, 1.2)
# plt.legend(loc='best')
# plt.show()


#relu 函数
# y_relu = F.relu(x).data.numpy()
# plt.plot(x_np, y_relu, c='red', label='relu')
# plt.ylim(-1, 5)
# plt.legend(loc='best')
# plt.show()


#leaky relu 函数


#softplus函数
# y_softplus = F.softplus(x).data.numpy()
# plt.plot(x_np, y_softplus, c='red', label='softplus')
# plt.ylim(-0.2, 6)
# plt.legend(loc='best')
# plt.show()



#mish 函数
y_mish = (x*torch.tanh(F.softplus(x))).data.numpy()
plt.plot(x_np, y_mish, c='red', label='mish')
plt.ylim(-1, 5)
plt.xlim(-5, 5)
plt.legend(loc='best')
plt.show()
