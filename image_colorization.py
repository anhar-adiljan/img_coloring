from __future__ import print_function

'''
Tensorflow implemenation of Image colorization using Adversarial loss
'''

import tensorflow as tf
import numpy as np
import scipy as sp
from vgg16 import vgg16
from scipy.misc import imread, imresize
from skimage import io, color, transform

batch_size = 5
num_epochs = 20
learning_rate = 1e-2

class colornet:
	# Shape: gray_imgs ?x224x224x1, slic_img ?x224x224x3
	def __init__(self, gray_imgs, slic_imgs=None, vgg_weights=None):
		self.gray_imgs = gray_imgs
		self.slic_imgs = slic_imgs
		gray_vgg_input = tf.concat([self.gray_imgs, self.gray_imgs, self.gray_imgs],3)
		self.gray_vgg = vgg16(gray_vgg_input, weights=vgg_weights)
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
			bn_input = self.normalize(tf.concat([self.gray_imgs, self.gray_imgs, self.gray_imgs],3))
			kernel = tf.Variable(tf.truncated_normal([3,3,3,3], dtype=tf.float32, stddev=1e-1))
			temp = tf.nn.conv2d(tf.add(bn_input, self.upconv1_2), kernel, [1,1,1,1], padding='SAME')
			kernel = tf.Variable(tf.truncated_normal([3,3,3,2], dtype=tf.float32, stddev=1e-1))
			self.uv_output = tf.nn.conv2d(temp, kernel, [1,1,1,1], padding='SAME')
			
	def normalize(self, x):
		mean, variance = tf.nn.moments(x, [0,1,2])
		#return tf.nn.batch_norm_with_global_normalization(x, mean, variance)
		return tf.contrib.layers.batch_norm(x)

def network(gray, weights=None):
	return colornet(gray, vgg_weights=weights).uv_output

# Load pre-trained VGG weights
weights = 'vgg16_weights.npz'

# Construct Model
imgs_uv = tf.placeholder(tf.float32, [None, 224, 224, 2])
imgs = tf.placeholder(tf.float32, [None, 224, 224, 1])
net = network(imgs, weights)

# define loss and optimizer
loss = tf.nn.l2_loss(tf.subtract(net, imgs_uv))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Load the images
imageCollection = io.imread_collection('tiny-imagenet-200/train/n09193705/images/*.JPEG')
images = np.array([])
print (len(imageCollection))
#images = imresize(images, (224,224))
for i in range(len(imageCollection)):
	np.append(images,imresize(imageCollection[i], (224,224)))
#images = images.concatenate()
n = len(images)
print(images.shape)
print(images[0,22,24,0:3])
# Convert the images into LUV color space
images_luv = color.rgb2luv(images)
images_l = images_luv[:,:,:,0]
# images_l -- to feed in imgs
# images_uv -- to feed in imgs_uv
images_l = images_l.reshape(images_l.shape + (1,))
images_uv = images_luv[:,:,:,1:3]

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	tr_losses = []
	print("Training begin")
	for epoch in range(num_epochs):
		# print("Epoch:", '%02d' % (epoch+1))
		avg_cost = 0.
		total_batch = int(n/batch_size)
		for i in range(total_batch):
			batch_l = images_l[i*batch_size:(i+1)*batch_size]
			batch_uv = images_uv[i*batch_size:(i+1)*batch_size]
			_, c = sess.run([optimizer, loss], feed_dict={imgs: batch_l, imgs_uv: batch_uv})
			# Compute average cost
			avg_cost += c / total_batch
			# print("Number of batches finished:", '%02d' % (i+1))
		avg_cost/=(224*224)
		tr_losses.append(avg_cost)
		out_uv =  sess.run(net, feed_dict={imgs: [images_l[0]]})
		out_img = color.luv2rgb(np.concatenate((images_l[0], out_uv[0]), 2))
		filename = 'result' + str(epoch) + '.png'
		io.imsave(filename, out_img)
		print("Epoch:", '%02d' % (epoch+1), "Training loss:", "{:.3f}".format(avg_cost))
	print("Optimization finished!")

'''
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
'''
		
		