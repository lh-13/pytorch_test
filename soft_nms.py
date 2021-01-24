'''
@version        : 1.0
@Author         : lh-13
@Date           : 2021-01-18 15:02:20
@LastEditors    : lh-13
@LastEditTime   : 2021-01-18 15:02:20
@Descripttion   : nms的又一实现
@FilePath       : /pytorch_test/nms_1.py
'''

'''
soft_nms
nms可以解决大部分重叠问题，但有些情况就无法解决
soft_nms 思路：
不要简单粗暴地删除所有IOU大于阈值的框，而是降低其置信度

nms 可以描述如下：将IOU大于阈值的窗口的得分全部置0。
softnms 改进有两种形式：

'''

import numpy as np  
import json 
import cv2  
from matplotlib import pyplot as plt
from copy import deepcopy

def show_img(imgs, dict1, dict2, tansed_name):
    #读取图片  
    image = cv2.imread(imgs)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_roi = deepcopy(image)

    for key, values in dict1.items():
        for det in values:
            score = det[4]
            det = list(map(int, det))
            cv2.rectangle(image_roi, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 8)
            cv2.putText(image_roi, str(score), (det[0], det[3]), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)

    image_nms = deepcopy(image)
    for key, values in dict2.items():
        for det in values:
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

def soft_nms(dicts, nms_threshold=0.5):
    if len(dicts) <= 0:
        return dicts  
    pickedBoxes = []   #最终返回值
    for label, boxes in dicts.items():
        #计算n个侯选框的面积大小
        boxes = np.array(boxes, dtype=np.float)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2-x1+1)*(y2-y1+1)

        #将scores进行排序，从大到小
        order_score_index = np.argsort(scores)[::-1]   #np.argsort 默认为升序排列，这里[::-1],逆序

        while (order_score_index.size > 0):
            box = boxes[order_score_index[0]]    #得分最高的那个框
            #用得分最高的那个框与后面其他框计算iou
            x11 = np.maximum(x1[order_score_index[0]], x1[order_score_index[1:]])
            y11 = np.maximum(y1[order_score_index[0]], y1[order_score_index[1:]])
            x22 = np.maximum(x2[order_score_index[0]], x2[order_score_index[1:]])
            y22 = np.maximum(y2[order_score_index[0]], y2[order_score_index[1:]])

            #计算交集区域
            w = np.maximum(x22-x11+1, 0.0)
            h = np.maximum(y22-y11+1, 0.0)
            intersection = w*h
            #计算iou
            iou = intersection / (areas[order_score_index[0]]+areas[order_score_index[1:]]-intersection)

            #进行阈值的判断
            for i in range(1, len(iou)+1):
                if iou[i-1] > nms_threshold:
                    #nms 直接删除， soft_nms对得分进行调整 
                    boxes[order_score_index[i]][4] *= 1 - iou[i-1]    #实现证明这个比下面的效果更好  #线性加权
                    # ov = 1-iou[i-1]
                    # sigma = 0.5
                    # boxes[order_score_index[i]][4] *= np.exp(-(ov*ov)/sigma)
            order_score_index = np.delete(order_score_index, [0])

        index = np.where(boxes[:, 4] > 0.5)
        boxes = boxes[index]

        dicts[label] = boxes  
        return dicts  







#if __name__ == '__main__':
    #predict_dict = {'cup': [[], [], [], [], [], []]}
imgPath = '2.jpg'
jsonPath = imgPath.replace('jpg', 'json')
dets = []
with open(jsonPath, encoding='utf_8') as f: 
    data = json.load(f)
    for item in data['shapes']:
        det = item['points'][0] + item['points'][1] + [item['label']]
        dets.append(det)

predict_dict = {'cup':np.array(dets,dtype=np.float)}
predict_dict_1 = predict_dict.copy()
predict_dict_2 = soft_nms(predict_dict, 0.5)
show_img(imgPath, predict_dict_1, predict_dict_2, 'After nms')
