""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------

#this is to check branch
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

sess=tf.InteractiveSession()
# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2_1 = 128 # 2nd layer number of neurons
n_hidden_2_2 = 128 # 2nd layer number of neurons
n_hidden_3_1 = 128 # 2nd layer number of neurons
n_hidden_3_2 = 128
n_hidden_3 = 256
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2_1': tf.Variable(tf.random_normal([128, n_hidden_2_1])),
    'h2_2': tf.Variable(tf.random_normal([128, n_hidden_2_2])),
    'h3_1': tf.Variable(tf.random_normal([128, 128])),
    'h3_2': tf.Variable(tf.random_normal([128,128])),
    'out': tf.Variable(tf.random_normal([256, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2_1': tf.Variable(tf.random_normal([n_hidden_2_1])),
    'b2_2': tf.Variable(tf.random_normal([n_hidden_2_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    tmp1=layer_1[:,:128]
    tmp2=layer_1[:,128:]
    
    layer_2_1 = tf.nn.relu(tf.add(tf.matmul(tmp1, weights['h2_1']), biases['b2_1']))
    layer_2_2 = tf.nn.relu(tf.add(tf.matmul(tmp2, weights['h2_2']), biases['b2_2']))
    
    
    layer_3_1 = tf.nn.relu(tf.matmul(layer_2_1, weights['h3_1']))
    layer_3_2 = tf.nn.relu(tf.matmul(layer_2_2, weights['h3_2']))
    
    merge=tf.concat([layer_3_1,layer_3_2],-1)+biases['b3']
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(merge, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


#with tf.Session() as sess:
sess.run(init)
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c ,acc= sess.run([train_op, loss_op,accuracy], feed_dict={X: batch_x,
                                                        Y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost),"acc={0}".format(acc))
print("Optimization Finished!")

# Test model


# Calculate accuracy

print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
