import tensorflow as tf
from os import listdir
import numpy as np
from skimage import novice

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

sess = tf.InteractiveSession()

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

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for j in range(20):
    np.random.shuffle(train_data)
    train_images=train_data[0:1500,0:27648]
    train_labels=np.zeros((1500,3))
    train_labels[np.arange(1500), np.reshape(train_data[0:1500,27648:27649].astype(int),[1500])]=1
    print 'Epoch '+str(j)
    for i in range(30):
      if i % 5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_images[(i*50):(i*50)+50,0:], y: train_labels[(i*50):(i*50)+50,0:], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: train_images[(i*50):(i*50)+50,0:], y: train_labels[(i*50):(i*50)+50,0:], keep_prob: 0.5})
    print('%d epoch test accuracy %g' % (j,accuracy.eval(feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})))

  print('final test accuracy %g' % accuracy.eval(feed_dict={x: test_images, y: test_labels, keep_prob: 1.0}))