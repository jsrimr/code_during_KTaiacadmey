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

#sess=tf.InteractiveSession()
#
#img=img.reshape(-1,28,28,1)
#W1=tf.Variable(tf.random_normal([3,3,1,5],std))