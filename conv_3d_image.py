'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-26 14:42:30
@LastEditors    : lh-13
@LastEditTime   : 2021-01-26 14:42:31
@Descripttion   : 读取image 实现3d卷积操作
@FilePath       : \pytorch_test\conv_3d_image.py
'''


import numpy as np 
import cv2   


#读取图片
img = cv2.imread('1.jpg')
cv2.imshow("src",img)
img_scale = cv2.resize(img, (64, 64))
cv2.imshow("src_scale",img_scale)

img_array = np.array(img_scale)
print(img_array.shape)
print(img_array)
#进行通道调整
img_array = np.transpose(img_array, (2, 0, 1))
print(img_array.shape)

#定义过滤器
filter = np.array([[[1, 0, -1],
                    [1, 0, -1], 
                    [1, 0, -1]],
                    [[1, 0, -1],
                    [1, 0, -1], 
                    [1, 0, -1]],
                    [[1, 0, -1],
                    [1, 0, -1], 
                    [1, 0, -1]],
                    ])

# filter = np.array([[[1, 0, -1],
#                     [1, 0, -1], 
#                     [1, 0, -1]],
#                     [[1, 1, 1],
#                     [0, 0, 0], 
#                     [-1, -1, -1]],
#                     [[1, 0, -1],
#                     [1, 0, -1], 
#                     [1, 0, -1]],
#                     ])
print(filter.shape)
print(filter)

#定义输出feature
feature = np.zeros((img_array.shape[0], img_array.shape[1]-filter.shape[1]+1, img_array.shape[2]-filter.shape[2]+1))
print(feature.shape)


#进行卷积操作
for c in range(0, feature.shape[0]):
    for row in range(0, feature.shape[1]):
        for col in range(0, feature.shape[2]):
            region = img_array[c, row:row+filter.shape[1], col:col+filter.shape[2]]
            product = region * filter[c, :, :]
            result = np.sum(product)

            feature[c, row, col] = result 

print(feature.shape)
print(feature)

result = np.transpose(feature, (1, 2, 0))
cv2.imshow("res", result)

cv2.waitKey(0)











