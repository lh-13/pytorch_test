'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-20 13:27:42
@LastEditors    : lh-13
@LastEditTime   : 2021-01-20 13:27:42
@Descripttion   : numpy 的一些方法
@FilePath       : \pytorch_test/np_test.py
'''

import numpy as np
import cv2     
import sys 

#-----------------------------------------------------------------使用numpy实现卷积操作

def conv_(input, cur_filter):
    filter_size = cur_filter.shape[0]
    result = np.zeros((input.shape))

    #循环遍历图像以应用卷积运算
    for row in np.uint16(np.arange(filter_size/2.0, input.shape[0]-filter_size/2.0+1)):     #+1 是因为np.arange[begin, end), 不包含end
        for col in np.uint16(np.arange(filter_size/2.0, input.shape[1]-filter_size/2.0+1)):
            #卷积的区域
            cur_region = input[row-np.uint16(np.floor(filter_size/2.0)):row+np.uint16(np.ceil(filter_size/2.0)), col-np.uint16(np.floor(filter_size/2.0)):col+np.uint16(np.ceil(filter_size/2.0))]
            #卷积操作
            cur_result = cur_region*cur_filter    
            conv_sum = np.sum(cur_result)
            #将求和保存到特征图中
            result[row, col] = conv_sum   

    #裁剪结果矩阵的异常值
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]

    return final_result    


def conv_myself(input, kernel=(3, 3), padding=0, stride=1):
    '''
    #input [n, c, h, w]
    #kernel [c, d, h, w]   #c为输入通道数，d为卷积的个数
    
    input[h, w, c]
    kernel[c, h, w]

    '''

    #判断图像通道数与卷积核对应关系   c==d 
    if len(input.shape) != 3 or len(kernel.shape) != 3:
        print('input or kernel shape has problem!')
        sys.exit()
    if input.shape[2] != kernel.shape[0]:
        print('input channel is not same as kernel input chanel')
        sys.exit()

    #检测过滤器是否是方阵
    if kernel.shape[1] != kernel.shape[2]:
        print('kernel height is not equal to weight')
        sys.exit() 
    
    #检测过滤器大小是否为奇数
    if kernel.shape[1] % 2 == 0:
        print('kernel height and weight must be even number')
        sys.exit()  

    #定义一个输出图   (输出大小：= (输入size + 2*padding - kernel.size)/stride)
    #feature_maps = np.zeros((input.shape[0], kernel.shape[1], (input.shape[2]+2*padding-kernel.shape[2])/stride, (input.shape[3]+2*padding-kernel.shape[3])/stride))
    feature_maps = np.zeros((input.shape[0] - kernel.shape[1]+1, input.shape[1]-kernel.shape[2]+1, kernel_filter.shape[0]))
    #进行卷积计算
    for filter_num in range(kernel.shape[0]):
        print('filter:', filter_num+1)
        cur_filter = kernel[filter_num, :, :]    #按通道划分卷积核，比如原卷积核为[2, 3, 3],这里需要拆分为2个[3,3],[3,3]

        #检测单个过滤器是否有多个通道。如果有，那么每个通道将对图像进行卷积。所有卷积的结果加起来得到一个特征图。
        if len(cur_filter.shape) > 2:
            continue   #temp

        else:
            #过滤器为2维，即单个通道
            conv_map = conv_(input, cur_filter)


    

    




    return 0





if __name__ == '__main__':
    image_path = '1.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640,640))
    #image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(image.shape)
    image_array = np.array(image)

    cv2.imshow('srcImage', image)

    #定义卷积核
    kernel = np.zeros([2, 3, 3])   #[d, h, w]    #如果是彩色图，则d=3
    #进行填充
    #检测垂直边缘
    kernel[0, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1], 
                                [-1, 0, 1]
                                ]])
    #检测水平边缘
    kernel[1, :, :] = np.array([[[1, 1, 1],
                                [0, 0, 0], 
                                [-1, -1, -1]
                                ]])
    print(kernel)

    feature_map = conv_myself(image, kernel)

    cv2.waitKey(0)



