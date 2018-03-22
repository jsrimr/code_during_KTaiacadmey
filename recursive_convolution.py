#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:18:19 2018

@author: ktai12
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess=tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


#convolution 압축을 확인하자!
#압축률이 다른 5장의 사진
img=mnist.train.images[0].reshape(28,28)

#original-28*28
plt.imshow(img)
plt.title('original')

W=tf.Variable(tf.random_normal([3,3,1,1],stddev=0.01))
img=img.reshape(-1,28,28,1)
conv2d=tf.nn.conv2d(img,W,strides=[1,1,1,1],padding="VALID")

sess.run(tf.global_variables_initializer())
conv2d_img1=conv2d.eval()
conv2d_img1.shape

#convolution once - 26*26
img1=conv2d_img1.reshape(26,26)
plt.imshow(img1)
plt.title('once')

#convolution twice

conv2d=tf.nn.conv2d(conv2d_img1,W,strides=[1,1,1,1],padding="VALID")
conv2d_img2=conv2d.eval()
conv2d_img2.shape

img2=conv2d_img2.reshape(24,24)
plt.imshow(img2)
plt.title('once')

tmp=img
for i in range(10):
    a=tmp.shape[0]
    tmp=tmp.reshape(-1,a,a,1)
    conv2d=tf.nn.conv2d(tmp,W,strides=[1,1,1,1],padding="VALID")
    conv2d_img=conv2d.eval()
    k=conv2d_img.shape[1]
    
    tmp=conv2d_img.reshape(k,k)
    plt.imshow(tmp)
    plt.title("{0}*{0} size".format(k))
    plt.show()