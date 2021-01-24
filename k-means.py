'''
@version        : 1.0
@Author         : lh-13
@Date           : 2020-12-31 15:27:35
@LastEditors    : lh-13
@LastEditTime   : 2020-12-31 15:27:36
@Descripttion   : 自己实现的k-means算法
@FilePath       : /pytorch_test/k-means.py
'''

import os  
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import random 

'''
k-means 聚类算法  (属于无监督学习)
思想：kmeans算法又名K均值算法，k-means算法中的k表示的是聚类为k个族，means代表取每一个聚类中数据的均值作为该簇的中心，或者称为质心，即用每一个的类的质心对该族进行描述
算法思想：先从样本集中随机选取k个样本作为簇中心，并计算所有样本与这k个簇中心的距离，对于每一个样本，将其划分到与其距离最近的“簇中心”的所在的簇中，对于新的簇计算新的“簇中心”
1.簇个数k的选择,并随机的选择k个点作为“簇中心”
2.计算各个样本点到“簇中心”的距离，并分为对应的k个簇
3.根据新划分的簇，更新“簇中心”
4.重复上述2、3过程，直至“簇中心”没有移动
'''

def CalcDistance(dataset, centroids, k):
    clalist = []
    for data in dataset:
        diff = np.tile(data, (k, 1)) - centroids  #(np.tile(a,(2,1)), 假设a=[0, 1, 2]就是先把a先沿x轴复制1倍，即没有复制，仍然是[0,1,2]。再把结果沿y方向复制2倍得到array([0,1,2],[0,1,2]))
        squareDiff = diff **2 #平方
        squareDist = np.sum(squareDiff, axis=1) #和(axis=1表示行)
        distance = squareDist ** 0.5   #开根号
        clalist.append(distance)
    clalist = np.array(clalist)    #返回一个每个点到质心的距离len(dataset)*k的数组

    return clalist  



def classify(dataset, centroids, k):
    #计算样本到质心的距离
    clalist = CalcDistance(dataset, centroids, k)
    #分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)   #asix=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataset).groupby(minDistIndices).mean()   #DataFramte(dataset)对dataset分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    #计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids



def kmeans(dataset, k):
    #随机选取质心
    centroids = random.sample(dataset, k)    #随机的从指定列表中取k个不同的元素(特别大的数据集可能比较慢)

    #更新质心，直到变化量全为0
    changed, newCentroids = classify(dataset, centroids, k)

    while np.any(changed != 0):
        changed, newCentroids = classify(dataset, newCentroids, k)

    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序

    #根据质心计算每个集群
    cluster = []
    clalist = CalcDistance(dataset, centroids, k)     #最后根据质心对数据进行分类
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  
        cluster[j].append(dataset[i])

    return centroids, cluster     

if __name__ == '__main__':
    k = 2
    #随机的产生一些二维的点并显示
    points_x = []
    points_y = []
    points = []
    for i in range(100):
        point = np.array([np.random.randint(1,100), np.random.randint(1,100)])
        points.append(point)
        points_x.append(point[0])
        points_y.append(point[1])


    plt.scatter(points_x, points_y)   #创建散点图

    #创建第二个散点图（主要为显示中心点）
    center1 = [20, 20]
    center2 = [80, 80]
    plt.scatter(5, 10, color='red')
    plt.show() 

    #开始k-means聚类
    centroids, cluster = kmeans(points, 2)
    print('质心为:%s' % centroids)
    print('集群为:%s' % cluster)
    for i in range(len(points)):
        plt.scatter(points[i][0], points[i][1], marker='o', color='green', label='原始点')
        
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()   



    
    


