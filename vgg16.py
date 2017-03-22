#########################################################################################
# COGS 181 Final Project                                                                #
# Neural network implementation in TensorFlow                                           #
# Author: Adilijiang Ainihaer (2017)                                                    #
#                                                                                       #
# Reference: https://www.cs.toronto.edu/~frossard/post/vgg16/                           #
# Author of reference: Davi Frossard (2016)                                             #
#                                                                                       #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md      #
#########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

class vgg16:
	def __init__(self, imgs, weights=None, sess=None):
		self.imgs = imgs
		self.sess = sess
		self.conv_layers()
		#if weights is not None and sess is not None:
			#self.load_weights(weights, sess)
		
	def conv_layers(self):
		self.parameters = []
		
		# zero-mean input (224x224x3)
		with tf.name_scope('preprocess') as scope:
			# This mean value is calculated over the entire ImageNet dataset
			mean = tf.constant([123.68,116.779,103.939], dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
			images = self.imgs - mean
		
		# conv1_1, 3x3 kernel, stride=[1,1,1,1], output 224x224x64
		with tf.name_scope('conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,3,64], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		
		# conv1_2, 3x3 kernel, stride=[1,1,1,1], output 224x224x64
		with tf.name_scope('conv1_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,64], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv1_2 = tf.nn.relu(out,name=scope)
			self.parameters += [kernel, biases]
		
		# pool1, output 112x112x64
		self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
		
		# conv2_1, 3x3 kernel, stride=[1,1,1,1], output 112x112x128
		with tf.name_scope('conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,64,128], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		
		# conv2_2, 3x3 kernel, stride=[1,1,1,1], output 112x112x128
		with tf.name_scope('conv2_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		
		# pool2, output 56x56x128
		self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
		
		# conv3_1, 3x3 kernel, stride=[1,1,1,1], output 56x56x256
		with tf.name_scope('conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,128,256], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv3_2, 3x3 kernel, stride=[1,1,1,1], output 56x56x256
		with tf.name_scope('conv3_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]

		# conv3_3, 3x3 kernel, stride=[1,1,1,1], output 56x56x256
		with tf.name_scope('conv3_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv3_2, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		
		# pool3, output 28x28x256
		self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
		
		# conv4_1, 3x3 kernel, stride=[1,1,1,1], output 28x28x512
		with tf.name_scope('conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,256,512], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
			
		# conv4_2, 3x3 kernel, stride=[1,1,1,1], output 28x28x512
		with tf.name_scope('conv4_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
			
		# conv4_3, 3x3 kernel, stride=[1,1,1,1], output 28x28x512
		with tf.name_scope('conv4_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3,3,512,512], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(self.conv4_2, kernel, [1,1,1,1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_3 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
			# perform batch normalization on self.conv4_3, 1x1 convolution, then upscale
			#kernel = tf.Variable(tf.truncated_normal([1,1,512,256], dtype=tf.float32, stddev=1e-1))
			#upconv4_3 = tf.nn.conv2d(normalize(self.conv4_3), kernel, [1,1,1,1], padding='SAME')
			#upconv4_3 = tf.image.resize_bilinear(upconv4_3, 
		
		# get hypercolumns: upscale and concatenate, output 224x224x963
		#upconv2_2 = tf.image.resize_bilinear(self.conv2_2, (224,224))
		#upconv3_3 = tf.image.resize_bilinear(self.conv3_3, (224,224))
		#upconv4_3 = tf.image.resize_bilinear(self.conv4_3, (224,224))
		#self.hypercolumns = tf.concat(3, [images, self.conv1_2, upconv2_2, upconv3_3, upconv4_3])
		#print tf.shape(self.hypercolumns)
	'''
	def normalize(self, x):
		mean, variance = tf.nn.moments(x, [0,1,2])
		return tf.nn.batch_norm_with_global_normalization(x, mean, variance)
	'''

'''
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16(imgs)
img1 = imread('laska.png', mode='RGB')
img1 = imresize(img1, (224, 224))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	hp = sess.run(vgg.hypercolumns, feed_dict={vgg.imgs: [img1]})
	print hp.shape
'''