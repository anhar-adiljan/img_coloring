from __future__ import print_function

'''
Tensorflow implemenation of Image colorization using Adversarial loss
'''

import tensorflow as tf
import numpy as np
import glob
import os
from vgg16 import vgg16
from scipy.misc import imread, imresize
from skimage import io, color 
from skimage.segmentation import slic

batch_size = 10
num_epochs = 200
learning_rate = 1e-3
display_step = 20

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

def write_imgs(out_imgs, epoch):
	filepath = 'test-output/epoch' + str(epoch) + '/'
	os.mkdir(filepath)
	for i in range(len(out_imgs)):
		io.imsave(filepath + 'test_out' + str(i) + '.png', out_imgs[i])

# Load pre-trained VGG weights
weights = 'vgg16_weights.npz'

# Construct Model
imgs_uv = tf.placeholder(tf.float32, [None, 224, 224, 2])
imgs = tf.placeholder(tf.float32, [None, 224, 224, 1])
net = network(imgs, weights)

# define loss and optimizer
loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(net, imgs_uv)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Load the images
print("Loading images...")
filepath = 'flower_photos/*/*.jpg'
tr_images = []
tst_images = []
tr_n = 2000
tst_n = 100
i = 0
j = 0
for filename in glob.glob(filepath):
	if i < tr_n:
		tr_image = imread(filename, mode='RGB')
		tr_image = imresize(tr_image, (224,224))
		tr_images.append(tr_image)
		i += 1
	elif j < tst_n:
		tst_image = imread(filename, mode='RGB')
		tst_image = imresize(tst_image, (224,224))
		tst_images.append(tst_image)
		j += 1
	else:
		break

tr_n = len(tr_images)
tr_images = np.array(tr_images)
tr_images = color.rgb2luv(tr_images)
tr_images_l = tr_images[:,:,:,0]
# images_l -- to feed in imgs
# images_uv -- to feed in imgs_uv
tr_images_l = tr_images_l.reshape(tr_images_l.shape + (1,))
tr_images_uv = tr_images[:,:,:,1:]

tst_n = len(tst_images)
tst_images = np.array(tst_images)
filepath = 'test-output/ref/'
os.mkdir(filepath)
for i in range(len(tst_images)):
	io.imsave(filepath + 'ref' + str(i) + '.png', tst_images[i])
tst_images = color.rgb2luv(tst_images)
tst_images_l = tst_images[:,:,:,0]
tst_images_l = tst_images_l.reshape(tst_images_l.shape + (1,))
tst_images_uv = tst_images[:,:,:,1:]

print("Training and test images loaded!")

# Apply SLIC algorithm on the training set
tr_images_segments = slic(tr_images)
print tr_images.shape

'''
print("Loading training set...")
#filepath = 'tiny-imagenet-200/val/images/*.JPEG'
filepath = 'helen_1/'
tr_images = []
tr_n = 1000
i = 0
for filename in glob.glob(filepath):
	if i < tr_n:
		tr_image = imread(filename, mode='RGB')
		tr_image = imresize(tr_image, (224,224))
		tr_images.append(tr_image)
		i += 1
	else:
		 break
tr_images = np.array(tr_images)
tr_images = color.rgb2luv(tr_images)
tr_images_l = tr_images[:,:,:,0]
# images_l -- to feed in imgs
# images_uv -- to feed in imgs_uv
tr_images_l = tr_images_l.reshape(tr_images_l.shape + (1,))
tr_images_uv = tr_images[:,:,:,1:]
print("Training set loaded!\n")

print("Loading test set...")
filepath = 'tiny-imagenet-200/test/images/*.JPEG'
tst_images = []
tst_n = 100
i = 0
for filename in sorted(glob.glob(filepath)):
	if i < tst_n:
		tst_image = imread(filename, mode='RGB')
		tst_image = imresize(tst_image, (224,224))
		tst_images.append(tst_image)
		i += 1
	else:
		break
tst_images = np.array(tst_images)
tst_images = color.rgb2luv(tst_images)
tst_images_l = tst_images[:,:,:,0]
tst_images_l = tst_images_l.reshape(tst_images_l.shape + (1,))
tst_images_uv = tst_images[:,:,:,1:]
print("Test set loaded!\n")
'''

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	tr_losses = []
	tst_losses = []
	print("Training begins...")
	for epoch in range(num_epochs):
		print("Epoch:", '%02d' % (epoch+1))
		avg_cost = 0.
		total_tr_batch = int(tr_n/batch_size)
		for i in range(total_tr_batch):
			batch_l = tr_images_l[i*batch_size:(i+1)*batch_size]
			batch_uv = tr_images_uv[i*batch_size:(i+1)*batch_size]
			_, c = sess.run([optimizer, loss], feed_dict={imgs: batch_l, imgs_uv: batch_uv})
			# Compute average cost
			avg_cost += c / total_tr_batch
			if (i+1) % display_step == 0:
				print("Number of batches finished:", '%02d' % (i+1))
		avg_cost /= (224*224)
		tr_losses.append(avg_cost)
		avg_cost = 0.
		print("Calculating test loss for epoch " + str(epoch))
		total_tst_batch = int(tst_n/batch_size)
		for i in range(total_tst_batch):
			batch_l = tst_images_l[i*batch_size:(i+1)*batch_size]
			batch_uv = tst_images_uv[i*batch_size:(i+1)*batch_size]
			c = sess.run(loss, feed_dict={imgs: batch_l, imgs_uv: batch_uv})
			# Compute average lost
			avg_cost += c / total_tst_batch
		avg_cost /= (224*224)
		tst_losses.append(avg_cost)
		print("Epoch:", '%02d' % (epoch+1), "Training loss:", "{:.3f}".format(tr_losses[-1]), "Test loss:", "{:.3f}".format(tst_losses[-1]))
		print("Writing test output to files...")
		out_uv = []
		for i in range(total_tst_batch):
			batch_l = tst_images_l[i*batch_size:(i+1)*batch_size]
			batch_out_uv =  sess.run(net, feed_dict={imgs: batch_l})
			out_uv.append(batch_out_uv)
		out_uv = np.concatenate(out_uv, axis=0)
		print(tst_images_l.shape)
		print(out_uv.shape)
		out_imgs_luv = np.concatenate((tst_images_l, out_uv), axis=3)
		out_imgs = []
		for i in range(len(out_imgs_luv)):
			out_imgs.append(color.luv2rgb(out_imgs_luv[i]))
		out_imgs = np.array(out_imgs)
		#print(out_imgs.shape)
		write_imgs(out_imgs, epoch)
		print("Output written to files!\n")
		# save training and test losses to file
		print("Writing current training and test losses to files\n")
		loss_filename = 'loss.txt'
		np.savetxt('tr_losses.txt', tr_losses)
		np.savetxt('tst_losses.txt', tst_losses)
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
		
		
