'''
@Author: lh-13
@Date: 2020-10-10 16:14:27
@LastEditTime: 2020-10-12 14:36:16
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pytorch_test\TorchvisionTransformsTest.py
'''
"""
torchvision.transforms  数据增加 test
author:lh-13
date:2020.0325
"""

import os 
import torch
import torchvision
from torchvision import transforms
from PIL import Image

outfile = './outfile'
img = Image.open('./cat.jpg')
#img.show()

#Resize 随机比例缩放
print(img.size)
# scale_img = transforms.Resize((100, 100))(img)
# print(scale_img.size)
# scale_img.save(os.path.join(outfile, 'scale_cat.jpg'))
# scale_img.show()

#随机位置裁剪
# random_crop_img = transforms.RandomCrop(100)(img)
# print(random_crop_img.size)
# random_crop_img.save(os.path.join(outfile, 'random_crop_img.jpg'))
# random_crop_img.show()
# random_crop_img_1 = transforms.RandomCrop((110, 200))(img)    #(h, w)
# print(random_crop_img_1.size)
# random_crop_img_1.save(os.path.join(outfile, 'random_crop_img_1.jpg'))
# random_crop_img_1.show()

#中心位置裁剪
# center_crop_img = transforms.CenterCrop((100, 200))(img)    #(h, w)
# print(center_crop_img.size)
# center_crop_img.save(os.path.join(outfile, 'center_crop_img.jpg'))
# center_crop_img.show()


#随机水平/垂直反转
# rand_h_flip_img = transforms.RandomHorizontalFlip(p=1)(img)   #p表示概率 default is 0.5
# print(rand_h_flip_img.size)
# rand_h_flip_img.save(os.path.join(outfile, 'rand_h_flip_img.jpg'))
# rand_h_flip_img.show()

# rand_v_flip_img = transforms.RandomVerticalFlip(p=1)(img)
# rand_v_flip_img.save(os.path.join(outfile, 'rand_v_flip_img.jpg'))
# rand_v_flip_img.show()

#随时旋转
# random_rotation = transforms.RandomRotation(45)(img)    #(-45, 45)之间随时旋转
# print(random_rotation.size)
# random_rotation.save(os.path.join(outfile, 'random_rotation.jpg'))
# random_rotation.show()

#亮度、对比度、饱和度、色度的变化
# brightness_img = transforms.ColorJitter(brightness=1)(img)
# brightness_img.save(os.path.join(outfile, 'brightness_img.jpg'))
# brightness_img.show()

# contrast_img = transforms.ColorJitter(contrast=1)(img)
# contrast_img.save(os.path.join(outfile, 'contrast_img.jpg'))
# contrast_img.show()

# saturation_img = transforms.ColorJitter(saturation=0.5)(img)
# saturation_img.save(os.path.join(outfile, 'saturation_img.jpg'))
# saturation_img.show()

# hue_img = transforms.ColorJitter(hue=0.5)(img)
# hue_img.save(os.path.join(outfile, 'hue_img.jpg'))
# hue_img.show()

#对图像进行随机遮挡    transforms.RandomErasing()  一定是对tensor进行的操作，需要先转成张量才能做
# image_tensor = transforms.ToTensor()(img)
# erasing_img_tensor = transforms.RandomErasing(p=0.5, value=1)(image_tensor)
# erasing_img = transforms.ToPILImage()(erasing_img_tensor)

# erasing_img.show()

#对图像进行仿射变换  transforms.RandomAffine(degrees, translate=None,scale=None, shear=None, resample=False,fillcolor=0)
affine_image = transforms.RandomAffine(degrees=30)(img)
affine_image.show()

#对图像进行随机旋转 transforms.RandomRotation(degrees)
rotation_image = transforms.RandomRotation(degrees=30)(img)
rotation_image.show()

