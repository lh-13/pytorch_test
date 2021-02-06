'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-26 13:30:58
@LastEditors    : lh-13
@LastEditTime   : 2021-01-26 13:30:58
@Descripttion   : 自己实现的2维卷积操作
@FilePath       : \pytorch_test\conv_2d.py
'''

import numpy as np   

#定义输出
input = np.array([[3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1], 
                [3, 1, 2, 2, 3], 
                [2, 0, 0, 2, 2], 
                [2, 0, 0, 0, 1],
                ])
print(input.shape)
print(input)

#定义过滤器
filter = np.array([[0, 1, 2],
                    [2, 2, 0], 
                    [0, 1, 2]])

print(filter.shape)
print(filter)


#定义输出
feature = np.zeros((input.shape[0]-filter.shape[0]+1, input.shape[1]-filter.shape[1]+1))
print(feature.shape)


#进行卷积操作
for row in range(0, feature.shape[0]):
    for col in range(0, feature.shape[1]):
        region = input[row:row+filter.shape[0],col:col+filter.shape[1]]
        product = region * filter 
        result = np.sum(product)

        feature[row, col] = result

print(feature.shape)
print(feature)



