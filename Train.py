import os
import cv2
import glob

import numpy as np
import tensorflow as tf

from Define import *
from Style_Utils import *
from VGG19 import *

# 1. dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content_path', dest='content_path', help='content image path', default='./content_image/SSAL.jpg', type=str)
    parser.add_argument('--style_path', dest='style_path', help='style image path', default='./style_image/kadinsky.jpg', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='save directory', default='./results_SSAL_kadinsky/', type=str)
    args = parser.parse_args()
    return args

args = parse_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

content_image = cv2.imread(args.content_path)
content_image = cv2.resize(content_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

style_image = cv2.imread(args.style_path)
style_image = cv2.resize(style_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

# cv2.imshow('content_image', content_image)
# cv2.imshow('style_image', style_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2. build
content_image = content_image.astype(np.float32) / 255.
content_image_var = tf.Variable(content_image, dtype = tf.float32, name = 'content')

style_image = style_image.astype(np.float32) / 255.
style_image_var = tf.Variable(style_image, dtype = tf.float32, name = 'style')

train_image = content_image.copy()
train_image_var = tf.Variable(train_image, dtype = tf.float32, name = 'train')

train_image_op = tf.clip_by_value(train_image_var * 255., clip_value_min = 0.0, clip_value_max = 255.)
train_image_op = tf.cast(train_image_op, dtype = tf.uint8)

_, content_end_points = vgg_19([content_image_var], reuse = False)
_, style_end_points = vgg_19([style_image_var], reuse = True)
_, train_end_points = vgg_19([train_image_var], reuse = True)

content_layer_dic = {'vgg_19/conv5/conv5_2' : {}}
style_layer_dic = {'vgg_19/conv1/conv1_1' : {},
                   'vgg_19/conv2/conv2_1' : {},
                   'vgg_19/conv3/conv3_1' : {},
                   'vgg_19/conv4/conv4_1' : {},
                   'vgg_19/conv5/conv5_1' : {}}
train_layer_dic = {}

content_ops = []

for name in content_end_points.keys():
    if name in content_layer_dic.keys():
        content_layer_dic[name]['content'] = content_end_points[name]

for name in style_end_points.keys():
    if name in style_layer_dic.keys():
        style_layer_dic[name]['style'] = gram_matrix(style_end_points[name])

for name in train_end_points.keys():
    if name in content_layer_dic.keys():
        content_layer_dic[name]['train'] = train_end_points[name]
    if name in style_layer_dic.keys():
        style_layer_dic[name]['train'] = gram_matrix(train_end_points[name])

# 3. loss function

# - style loss
style_loss_list = []
for name in style_layer_dic.keys():
    style_loss = tf.reduce_mean(tf.pow(style_layer_dic[name]['style'] - style_layer_dic[name]['train'], 2))
    style_loss_list.append(style_loss)

style_loss_op = tf.add_n(style_loss_list)
style_loss_op = STYLE_WEIGHT * style_loss_op / len(style_layer_dic.keys())

# - content loss
content_loss_list = []
for name in content_layer_dic.keys():
    content_loss = tf.reduce_mean(tf.pow(content_layer_dic[name]['content'] - content_layer_dic[name]['train'], 2))
    content_loss_list.append(content_loss)

content_loss_op = tf.add_n(content_loss_list)
content_loss_op = CONTENT_WEIGHT * content_loss_op / len(content_layer_dic.keys())

# - total loss
loss_op = style_loss_op + content_loss_op

# 4. train
train_op = tf.train.AdamOptimizer(learning_rate = INIT_LEARNING_RATE, beta1 = 0.99, epsilon = 1e-1).minimize(loss_op, var_list = [train_image_var])

# 5. session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

vgg_vars = []
for var in tf.trainable_variables():
    if 'vgg' in var.name:
        vgg_vars.append(var)

saver = tf.train.Saver(var_list = vgg_vars)
saver.restore(sess, './vgg19/vgg_19.ckpt')

for epoch in range(MAX_EPOCHS + 1):
    _, loss, style_loss, content_loss, result_image = sess.run([train_op, loss_op, style_loss_op, content_loss_op, train_image_op])
    
    cv2.imshow('show', result_image)
    cv2.waitKey(1)

    if epoch % 10 == 0:
        cv2.imwrite('{}{}.jpg'.format(args.save_dir, epoch), result_image.astype(np.uint8))
        print('[{}/{}] loss = {:.0f}, style_loss = {:.0f}, content_loss = {:.0f}'.format(epoch, MAX_EPOCHS, loss, style_loss, content_loss))

    
