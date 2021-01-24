'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-07 10:15:44
@LastEditors    : lh-13
@LastEditTime   : 2021-01-07 10:15:45
@Descripttion   : 自己实现的knn算法(k近邻)
@FilePath       : /pytorch_test/knn.py
'''

'''
6个训练样本，分为三类，每个样本有4个特征，编号7为我们要预测的

编号    花萼长度(cm)    花萼宽度(cm)    花瓣长度(cm)    花瓣宽度(cm)    名称
1           4.9             3.1             1.5             0.1         Iris setosa
2           5.4             3.7             1.5             0.2         Iris setosa
3           5.2             2.7             3.9             1.4         Iris versicolor
4           5.0             2.0             3.5             1.0         Iris versicolor
5           6.3             2.7             4.9             1.8         Iris virginica
6           6.7             3.3             5.7             2.1         Iris virginica
7           5.5             2.5             4.0             1.3         ？
'''

#计算测试样本到各个训练样本的距离

import numpy as np  

def CalcDistance(listA, listB):
    diff = listA - listB    #减
    squareDiff = diff**2     #平方
    squareDist = np.sum(squareDiff) #和(axis=1表示行)
    distance = squareDist ** 0.5   #开根号

    return distance 


if __name__ == '__main__':
    testData = np.array([5.5, 2.5, 4.0, 1.3])
    print("Distance to 1:", CalcDistance(np.array([4.9, 3.1, 1.5, 0.1]), testData))
    print("Distance to 2:", CalcDistance(np.array([5.4, 3.7, 1.5, 0.2]), testData))
    print("Distance to 3:", CalcDistance(np.array([5.2, 2.7, 3.9, 1.4]), testData))
    print("Distance to 4:", CalcDistance(np.array([5.0, 2.0, 3.5, 1.0]), testData))
    print("Distance to 5:", CalcDistance(np.array([6.3, 2.7, 4.9, 1.8]), testData))
    print("Distance to 6:", CalcDistance(np.array([6.7, 3.3, 5.7, 2.1]), testData))
    

