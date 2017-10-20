import numpy as np 
import matplotlib as mp
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
from os import listdir
from skimage import novice
from scipy.misc import imshow
from scipy.ndimage import rotate

d_images = list()
u_images = list()
p_images = list()
for image in listdir('full_dataset/d'):
    d_images.append('full_dataset/d/'+image)
for image in listdir('full_dataset/u'):
    u_images.append('full_dataset/u/'+image)
for image in listdir('full_dataset/p'):
    p_images.append('full_dataset/p/'+image)
data = np.ndarray([1986,96*96*3])
i = 0
for image in d_images:
    raw = novice.open(image)
    raw.size = (96,96)
    data[i] = raw.rgb.reshape([96*96*3])
    i = i+1
for image in u_images:
    raw = novice.open(image)
    raw.size = (96,96)
    data[i] = raw.rgb.reshape([96*96*3])
    i = i+1
for image in p_images:
    raw = novice.open(image)
    raw.size = (96,96)
    data[i] = raw.rgb.reshape([96*96*3])
    i = i+1
data = np.hstack([data, np.zeros([1986,1])])
for i in range(1372,1774):
    data[i][27648]=1
for i in range(1774,1986):
    data[i][27648]=2

np.random.shuffle(data)
train_data = data[0:1500,0:27649]
test_data = data[1500:1986,0:27649]

tf.reset_default_graph()

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, shape=[None, 27648])
y = tf.placeholder(tf.float32, shape=[None, 3])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 96, 96, 3])
W_conv1 = weight_variable([7, 7, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1, 12*12*128])
W_fc1 = weight_variable([12 * 12 * 128, 512])
b_fc1 = bias_variable([512])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([512, 3])
b_fc3 = bias_variable([3])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_images=test_data[0:486,0:27648]
test_labels=np.zeros((486,3))
test_labels[np.arange(486), np.reshape(test_data[0:486,27648:27649].astype(int),[486])]=1

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for j in range(2):
    np.random.shuffle(train_data)
    train_images=train_data[0:1500,0:27648]
    train_labels=np.zeros((1500,3))
    train_labels[np.arange(1500), np.reshape(train_data[0:1500,27648:27649].astype(int),[1500])]=1
    print 'Epoch '+str(j)
    for i in range(30):
        sess.run(train_step, feed_dict={x: train_images[(i*50):(i*50)+50,0:], y: train_labels[(i*50):(i*50)+50,0:], keep_prob: 1.0})
        if (i+1) % 5 == 0:
            trainAccuracy = sess.run(accuracy, feed_dict={x: train_images[(i*50):(i*50)+50,0:], y: train_labels[(i*50):(i*50)+50,0:], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i+1, trainAccuracy))

testAccuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob:1.0})
print("test accuracy %g"%(testAccuracy))

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,27648],order='F'),keep_prob:1.0})
    plotNNFilter(units)
def plotNNFilter(units):
    print(units.shape[3])
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(rotate(units[0,:,:,i],90))

imageToUse = test_images[0]
plt.imshow(rotate(np.reshape(imageToUse,[96,96,3]),90))

getActivations(h_conv1,imageToUse)

getActivations(h_conv2,imageToUse)

getActivations(h_conv3,imageToUse)