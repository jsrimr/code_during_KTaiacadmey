#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:21:43 2018

@author: ktai12
"""

#%matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess=tf.InteractiveSession()

image=np.array([[[[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]]]],dtype=np.float32)

print(image.shape)
plt.imshow(image.reshape(3,3),cmap='Greys')

weight=tf.constant([[[[1.,-10.,-1.]],[[1.,-10.,-1.]]],
                    [[[1.,-10.,-1.]],[[1.,-10.,-1.]]]])

print("weight.shape",weight.shape)

#VALID-> 0패딩 안씀. 이미지가 줄어드는 한이 있어도 유효하게 가져가겠다. #SAME도 있음 

conv2d=tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="SAME") 

conv2d_img=conv2d.eval()
print("conv2d_img.shape",conv2d_img.shape)

conv2d_img=np.swapaxes(conv2d_img,0,3)

for i,one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1)  
    plt.imshow(one_img.reshape(3,3),cmap='gray')