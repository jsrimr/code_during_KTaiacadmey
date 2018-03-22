#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:08:22 2018

@author: ktai12
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

img=mnist.train.images[0].reshape(28,28)
plt.imshow(img,cmap="gray")

sess=tf.InteractiveSession()

img=img.reshape(-1,28,28,1) #몇 장 가져오는지 모를때
W1=tf.Variable(tf.random_normal([3,3,1,5],stddev=0.01))
conv2d=tf.nn.conv2d(img,W1,strides=[1,1,1,1],padding="SAME")
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img=conv2d.eval()
print("Before size : ",conv2d_img.shape)
conv2d_img=np.swapaxes(conv2d_img,0,3)
print("After size : ",conv2d_img.shape)

for i,one_img in enumerate(conv2d_img):
    print("one_img shape= : ", one_img.shape)
    plt.subplot(2,3,i+1),plt.imshow(one_img.reshape(28,28),cmap='gray')