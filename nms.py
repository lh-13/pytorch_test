'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-13 10:35:34
@LastEditors    : lh-13
@LastEditTime   : 2021-01-13 10:35:35
@Descripttion   : nms自己算法实现
@FilePath       : /pytorch_test/nms.py
'''

'''
Non-Maximum Suppression 非极大值抑制
思想：搜索局部极大值，抑制非极大值元素。广泛应用于目标检测算法中，其目的是为了消除多余的候选框，找到最佳的物体检测位置

现，假设有一个侯选boxes的集合B和其对应的scores集合S
1.找出分数最高的那个框M
2.将M对应的Box从B中删除
3.将删除的box添加到集合D中
4.从B中删除与M对应的Box重叠区域大于阈值threshold的其他框
5.重复上述步骤1到4  
'''

import numpy as np   
import json   
from matplotlib import pyplot as plt 
import cv2  
from copy import deepcopy

def show_img(imgs, dets1, dets2, tansed_name):
    #读取图片  
    image = cv2.imread(imgs)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_roi = deepcopy(image)

    for det in dets1:
        score = det[4]
        det = list(map(int, det))
        cv2.rectangle(image_roi, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 8)
        cv2.putText(image_roi, str(score), (det[0], det[3]), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)

    image_nms = deepcopy(image)
    for det in dets2:
        score = det[4]
        det = list(map(int, det))
        cv2.rectangle(image_nms, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 8)
        cv2.putText(image_nms, str(score), (det[0], det[3]), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)
    
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image_roi)
    plt.title("Origin")

    plt.subplot(1, 2, 2)
    plt.imshow(image_nms)
    plt.title(tansed_name)

    plt.show()  






def nms(boundingBoxes, nmsThreshold = 0.5):
    
    if len(boundingBoxes) == 0:
        return [], []
    
    bboxes = np.array(boundingBoxes)
    #计算n个侯选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2-x1+1)*(y2-y1+1)  #+1 可加可不加

    #对置信度进行排序，获取排序后的下标序号，argsort 默认从小到大排序
    orderScore = np.argsort(scores)

    pickedBoxes = []   #最终返回值

    while (orderScore.size > 0):
        #将置信度最大的框加入返回值列表中
        bigScoreIndex = orderScore[-1]
        pickedBoxes.append(boundingBoxes[bigScoreIndex])

        #获取当前置信度最大的侯选框与其他候选框的相交面积
        x11 = np.maximum(x1[bigScoreIndex], x1[orderScore[:-1]])
        y11 = np.maximum(y1[bigScoreIndex], y1[orderScore[:-1]])
        x22 = np.minimum(x2[bigScoreIndex], x2[orderScore[:-1]])
        y22 = np.minimum(y2[bigScoreIndex], y2[orderScore[:-1]])

        w = np.maximum(0.0, x22-x11+1)
        h = np.maximum(0.0, y22-y11+1)
        intersection = w*h   

        #利用相交的面积和两个框自身的面积计算框的交并比，将交并比大于阈值的框删除
        ious = intersection / (areas[bigScoreIndex] + areas[orderScore[:-1]] - intersection)
        left = np.where(ious < nmsThreshold)
        orderScore = orderScore[left]

    return pickedBoxes


#if __name__ == '__main__':
imgPath = '1.jpg'
jsonPath = imgPath.replace('jpg', 'json')
dets = []
with open(jsonPath, encoding='utf_8') as f: 
    data = json.load(f)
    for item in data['shapes']:
        det = item['points'][0] + item['points'][1] + [item['label']]
        dets.append(det)

dets = np.array(dets, dtype=np.float)

dets2 = nms(dets, 0.5)
dets1 = dets  
show_img(imgPath, dets1, dets2, 'After nms')










