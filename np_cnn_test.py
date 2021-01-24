'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-21 14:08:48
@LastEditors    : lh-13
@LastEditTime   : 2021-01-21 14:08:48
@Descripttion   :  网上用numpy实现的卷积操作  https://www.cnblogs.com/chenxiangzhen/p/10384955.html
@FilePath       : /pytorch_test/np_cnn_test.py
'''

import numpy as np  
import sys  

import skimage.data
import matplotlib
import matplotlib.pyplot as plt


def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    # 循环遍历图像以应用卷积运算
    for r in np.uint16(np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1)):
            # 卷积的区域
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),
                          c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            # 卷积操作
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            # 将求和保存到特征图中
            result[r, c] = conv_sum

        # 裁剪结果矩阵的异常值
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0),
                   np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result


def conv(img, conv_filter):
    # 检查图像通道的数量是否与过滤器深度匹配
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("错误：图像和过滤器中的通道数必须匹配")
            sys.exit()

    # 检查过滤器是否是方阵
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('错误：过滤器必须是方阵')
        sys.exit()

    # 检查过滤器大小是否是奇数
    if conv_filter.shape[1] % 2 == 0:
        print('错误：过滤器大小必须是奇数')
        sys.exit()

    # 定义一个空的特征图，用于保存过滤器与图像的卷积输出
    feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                             img.shape[1] - conv_filter.shape[1] + 1,
                             conv_filter.shape[0]))

    # 卷积操作
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]

        # 检查单个过滤器是否有多个通道。如果有，那么每个通道将对图像进行卷积。所有卷积的结果加起来得到一个特征图。
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        else:
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map
    return feature_maps


def pooling(feature_map, size=2, stride=2):
    # 定义池化操作的输出
    pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride + 1),
                         np.uint16((feature_map.shape[1] - size + 1) / stride + 1),
                         feature_map.shape[-1]))

    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size + 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size + 1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r: r+size, c: c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out

def relu(feature_map):  
    #Preparing the output of the ReLU activation function.  
    relu_out = np.zeros(feature_map.shape)  
    for map_num in range(feature_map.shape[-1]):  
        for r in np.arange(0,feature_map.shape[0]):  
            for c in np.arange(0, feature_map.shape[1]):  
                relu_out[r, c, map_num] = np.max(feature_map[r, c, map_num], 0)
    
    return relu_out




# 读取图像
img = skimage.data.chelsea()
# 转成灰度图像
img = skimage.color.rgb2gray(img)

# 初始化卷积核
l1_filter = np.zeros((2, 3, 3))
# 检测垂直边缘
l1_filter[0, :, :] = np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
# 检测水平边缘
l1_filter[1, :, :] = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])

"""
第一个卷积层
"""
# 卷积操作
l1_feature_map = conv(img, l1_filter)
# ReLU
l1_feature_map_relu = relu(l1_feature_map)
# Pooling
l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)

"""
第二个卷积层
"""
# 初始化卷积核
l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
# 卷积操作
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
# ReLU
l2_feature_map_relu = relu(l2_feature_map)
# Pooling
l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)

"""
第三个卷积层
"""
# 初始化卷积核
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
# 卷积操作
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
# ReLU
l3_feature_map_relu = relu(l3_feature_map)
# Pooling
l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)

"""
结果可视化
"""
fig0, ax0 = plt.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
plt.savefig("in_img1.png", bbox_inches="tight")
plt.close(fig0)

# 第一层
fig1, ax1 = plt.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")

ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")

ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")

plt.savefig("L1.png", bbox_inches="tight")
plt.close(fig1)

# 第二层
fig2, ax2 = plt.subplots(nrows=3, ncols=3)
ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])
ax2[0, 0].set_title("L2-Map1")

ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])
ax2[0, 1].set_title("L2-Map2")

ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
ax2[0, 2].get_xaxis().set_ticks([])
ax2[0, 2].get_yaxis().set_ticks([])
ax2[0, 2].set_title("L2-Map3")

ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])
ax2[1, 0].set_title("L2-Map1ReLU")

ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])
ax2[1, 1].set_title("L2-Map2ReLU")

ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
ax2[1, 2].get_xaxis().set_ticks([])
ax2[1, 2].get_yaxis().set_ticks([])
ax2[1, 2].set_title("L2-Map3ReLU")

ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])
ax2[2, 0].set_title("L2-Map1ReLUPool")

ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])
ax2[2, 1].set_title("L2-Map2ReLUPool")

ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
ax2[2, 2].get_xaxis().set_ticks([])
ax2[2, 2].get_yaxis().set_ticks([])
ax2[2, 2].set_title("L2-Map3ReLUPool")

plt.savefig("L2.png", bbox_inches="tight")
plt.close(fig2)

# 第三层
fig3, ax3 = plt.subplots(nrows=1, ncols=3)
ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
ax3[0].get_xaxis().set_ticks([])
ax3[0].get_yaxis().set_ticks([])
ax3[0].set_title("L3-Map1")

ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
ax3[1].get_xaxis().set_ticks([])
ax3[1].get_yaxis().set_ticks([])
ax3[1].set_title("L3-Map1ReLU")

ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax3[2].get_xaxis().set_ticks([])
ax3[2].get_yaxis().set_ticks([])
ax3[2].set_title("L3-Map1ReLUPool")

plt.savefig("L3.png", bbox_inches="tight")
plt.close(fig3)


