from __future__ import print_function

'''
Tensorflow implemenation of Image colorization using Adversarial loss
'''

import tensorflow as tf
import numpy as np
import scipy as sp
from vgg16 import vgg16
from scipy.misc import imread, imresize
from skimage import io, color

class colornet:
	# Shape: gray_imgs ?x224x224x1, slic_img ?x224x224x3
	def __init__(self, gray_imgs, slic_imgs=None, vgg_weights=None, sess=None):
		self.gray_imgs = gray_imgs
		self.slic_imgs = slic_imgs
		self.sess = sess
		gray_vgg_input = tf.concat(3, [self.gray_imgs, self.gray_imgs, self.gray_imgs])
		self.gray_vgg = vgg16(gray_vgg_input, weights=vgg_weights, sess=sess)
		self.uplayers()
	
	def uplayers(self):
		with tf.name_scope('conv4_3') as scope:
			#perform batch normalization on self.gray_vgg.conv4_3, 1x1 convolution, then upscale
			kernel = tf.Variable(tf.truncated_normal([1,1,512,256], dtype=tf.float32, stddev=1e-1))
			# input 28x28x512; output 56x56x256
			self.upconv4_3 = tf.nn.conv2d(self.normalize(self.gray_vgg.conv4_3), kernel, [1,1,1,1], padding='SAME')
			self.upconv4_3 = tf.image.resize_bilinear(self.upconv4_3, (56,56))
		
		with tf.name_scope('conv3_3') as scope:
			#perform batch normalization on self.gray_vgg.conv3_3, 3x3 convolution, then upscale
			bn_conv3_3 = self.normalize(self.gray_vgg.conv3_3)
			kernel = tf.Variable(tf.truncated_normal([3,3,256,128], dtype=tf.float32, stddev=1e-1))
			# input 56x56x256; output 112x112x128
			self.upconv3_3 = tf.nn.conv2d(tf.add(bn_conv3_3, self.upconv4_3), kernel, [1,1,1,1], padding='SAME')
			self.upconv3_3 = tf.image.resize_bilinear(self.upconv3_3, (112,112))
			
		with tf.name_scope('conv2_2') as scope:
			#perform batch normalization on self.gray_vgg.conv2_2, 3x3 convolution, then upscale
			bn_conv2_2 = self.normalize(self.gray_vgg.conv2_2)
			kernel = tf.Variable(tf.truncated_normal([3,3,128,64], dtype=tf.float32, stddev=1e-1))
			# input 112x112x128; output 224x224x64
			self.upconv2_2 = tf.nn.conv2d(tf.add(bn_conv2_2, self.upconv3_3), kernel, [1,1,1,1], padding='SAME')
			self.upconv2_2 = tf.image.resize_bilinear(self.upconv2_2, (224,224))

		with tf.name_scope('conv1_2') as scope:
			#perform batch normalization on self.gray_vgg.conv1_2, 3x3 convolution, no upscale
			bn_conv1_2 = self.normalize(self.gray_vgg.conv1_2)
			kernel = tf.Variable(tf.truncated_normal([3,3,64,3], dtype=tf.float32, stddev=1e-1))
			# input 224x224x64; output 224x224x3
			self.upconv1_2 = tf.nn.conv2d(tf.add(bn_conv1_2, self.upconv2_2), kernel, [1,1,1,1], padding='SAME')
		
		with tf.name_scope('output') as scope:
			#perform batch normalization on gray_vgg_input, two 3x3 convolution
			bn_input = self.normalize(tf.concat(3, [self.gray_imgs, self.gray_imgs, self.gray_imgs]))
			kernel = tf.Variable(tf.truncated_normal([3,3,3,3], dtype=tf.float32, stddev=1e-1))
			temp = tf.nn.conv2d(tf.add(bn_input, self.upconv1_2), kernel, [1,1,1,1], padding='SAME')
			kernel = tf.Variable(tf.truncated_normal([3,3,3,2], dtype=tf.float32, stddev=1e-1))
			self.uv_output = tf.nn.conv2d(temp, kernel, [1,1,1,1], padding='SAME')
			
	def normalize(self, x):
		mean, variance = tf.nn.moments(x, [0,1,2])
		#return tf.nn.batch_norm_with_global_normalization(x, mean, variance)
		return tf.contrib.layers.batch_norm(x)

imgs = tf.placeholder(tf.float32, [None, 224, 224, 1])
net = colornet(imgs)
img1 = imread('laska.png', mode='RGB')
img1 = imresize(img1, (224, 224))
gray_input = color.rgb2gray(img1)
gray_rgb = color.gray2rgb(gray_input)
print(gray_rgb.shape)
io.imsave('gray.png', gray_rgb)
gray_input = gray_input.reshape(224,224,1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	uv_output = sess.run(net.uv_output, feed_dict={net.gray_imgs: [gray_input]})
	yuv_output = np.concatenate(([gray_input], uv_output), 3)
	for yuv in yuv_output:
		print(yuv)
		print(yuv.shape)
		rgb = color.luv2rgb(yuv)
		io.imsave('output.png', rgb)
		
		