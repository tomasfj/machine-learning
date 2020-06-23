########################################################################################################################
# CONVOLUTIONAL NEURAL NETWORKS (GTSRB)
########################################################################################################################

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import cv2
from random import shuffle, sample
import pickle
import sklearn.metrics as mt
import math
import datetime
import csv
import argparse
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')


# ##########################
# Configs

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='CSV dataset file')
ap.add_argument('-i', '--input_folder', required=True, help='Data input folder')
ap.add_argument('-o', '--output_folder', required=True, help='Results/debug output folder')
ap.add_argument('-b', '--batch_size', type=int, default=100, help='Learning batch size')
ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
ap.add_argument('-r', '--drop_out_rate', type=float, default=0.7, help='Drop out rate used in the deep model')
ap.add_argument('-e', '--epochs', type=int, default=10, help='Tot. epochs')
ap.add_argument('-p', '--patience', type=int, default=0, help='Maximum number of consecutive iterations increasing loss')

args = vars(ap.parse_args())

if not os.path.isdir(args['output_folder']):
    os.mkdir(args['output_folder'])

date_time_folder = os.path.join(args['output_folder'], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir(date_time_folder)

imdb_file = args['dataset'][:-4] + '.dat'

lr_decay = 0.9
decay_epochs = 100

COLORS = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:cyan']

# #####################################################################################################################
# Load Data in '.csv' format: filename, class, learning_validation_test_set_flag


# #######################################
# Divide data into learning | test samples
# #######################################
learning_samples = []
test_samples = []
validation_samples = []
with open(args['dataset']) as f:
    csv_file = csv.reader(f, delimiter=',')
    for row in csv_file:
        if int(row[-1]) == 0:
            learning_samples.append(row[:-1])
        else:
            if int(row[-1]) == 1:
                validation_samples.append(row[:-1])
            else:
                test_samples.append(row[:-1])

for el in learning_samples:
    el[-1] = int(el[-1])

for el in validation_samples:
    el[-1] = int(el[-1])

for el in test_samples:
    el[-1] = int(el[-1])

shuffle(learning_samples)
shuffle(validation_samples)
shuffle(test_samples)


# #######################################
# Create "learning_classes" | "validation_classes" | "test_classes" information
# #######################################


learning_classes = []
for row in learning_samples:
    row[0] = os.path.join(args['input_folder'], row[0])
    learning_classes.append(row[-1])

validation_classes = []
for row in validation_samples:
    row[0] = os.path.join(args['input_folder'], row[0])
    validation_classes.append(row[-1])

test_classes = []
for row in test_samples:
    row[0] = os.path.join(args['input_folder'], row[0])
    test_classes.append(row[-1])


classes = np.unique(np.asarray(learning_classes + validation_classes + test_classes), axis=0)


onehot_encoder = OneHotEncoder(sparse=False)
learning_classes = onehot_encoder.fit_transform(np.reshape(np.asarray(learning_classes), (-1, 1)))
validation_classes = onehot_encoder.fit_transform(np.reshape(np.asarray(validation_classes), (-1, 1)))
test_classes = onehot_encoder.fit_transform(np.reshape(np.asarray(test_classes), (-1, 1)))

#####################################################################################################

# get the image size
img = cv2.imread(learning_samples[0][0], cv2.IMREAD_COLOR)
HEIGHT_IMGS = 64
WIDTH_IMGS = 64
DEPTH_IMGS = np.size(img, 2)


############################
# AUXILIARY FUNCTIONS

def read_batch(l_s, l_s_classes, rand_idx, imd):
    imgs = np.zeros((len(rand_idx), HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS), dtype=np.float32)
    cls = np.zeros((len(rand_idx), len(classes)), dtype=np.float32)

    for i, idx in enumerate(rand_idx):
        imgs[i, :, :, :] = cv2.resize(cv2.imread(l_s[idx][0], cv2.IMREAD_COLOR), (HEIGHT_IMGS, WIDTH_IMGS))
        cls[i, :] = l_s_classes[idx, :]

    for i, _ in enumerate(rand_idx):
        imgs[i, :, :, :] = np.divide(imgs[i, :, :, :], imd)
    return imgs, cls


def get_imdb(paths):
    imdb = np.zeros((HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS), dtype=np.float32)
    for i, pt in enumerate(paths):
        #print('IMDB: \{ \}/ \{ \}'.format(i, len(paths)))
        img = cv2.resize(cv2.imread(pt[0], cv2.IMREAD_COLOR), (HEIGHT_IMGS, WIDTH_IMGS))
        imdb = imdb + img
    imdb = imdb / len(paths)
    return imdb


########################################################################################################################
# TENSORFLOW
########################################################################################################################


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float", initializer=tfc.layers.xavier_initializer())
        b = tf.get_variable("b", [outputD], dtype = "float", initializer=tfc.layers.xavier_initializer())
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME"):
    """convolutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum], initializer=tfc.layers.xavier_initializer())
        b = tf.get_variable("b", shape = [featureNum], initializer=tfc.layers.xavier_initializer())
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
    return tf.nn.relu(out, name=scope.name)


def vgg_mini(x_inputs, keepPro, classNum):
    conv1_1 = convLayer(x_inputs, 3, 3, 1, 1, 32, "conv1_1" )
    conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 32, "conv1_2")
    pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

    conv2_1 = convLayer(pool1, 3, 3, 1, 1, 32, "conv2_1")
    conv2_4 = convLayer(conv2_1, 3, 3, 1, 1, 32, "conv2_4")
    pool2 = maxPoolLayer(conv2_4, 2, 2, 2, 2, "pool2")

    conv3_1 = convLayer(pool2, 3, 3, 1, 1, 32, "conv3_1")
    conv3_3 = convLayer(conv3_1, 3, 3, 1, 1, 32, "conv3_3")
    pool3 = maxPoolLayer(conv3_3, 2, 2, 2, 2, "pool3")

    fcIn = tf.reshape(pool3, [-1, 8*8*32])
    fc6 = fcLayer(fcIn, 8*8*32, 384, True, "fc6")
    drop1 = tf.nn.dropout(fc6, keepPro)

    fc7 = fcLayer(drop1, 384, 192, True, "fc7")
    drop2 = tf.nn.dropout(fc7, keepPro)

    fc8 = fcLayer(drop2, 192, classNum, False, "fc8")

    return fc8


###############################
# Create Placeholders

x_input_shape = (None, HEIGHT_IMGS, WIDTH_IMGS, DEPTH_IMGS)
x_inputs = tf.placeholder(tf.float32, shape=x_input_shape)
y_model = tf.placeholder(tf.float32, shape=(None, len(classes)))
y_targets = tf.placeholder(tf.float32, shape=(None, len(classes)))
generation_num = tf.Variable(0, trainable=False)

def loss_cross_entropy(logits, targets):
    lo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
    return lo


def loss_l2(logits, targets):
    lo = tf.losses.mean_squared_error(labels=targets, logits=logits)
    return lo


with tf.variable_scope('model_definition') as scope:
    model_outputs = vgg_mini(x_inputs, args['drop_out_rate'], len(classes))
    scope.reuse_variables()

loss = loss_cross_entropy(model_outputs, y_targets)

model_learning_rate = tf.train.exponential_decay(args['learning_rate'], generation_num, decay_epochs, lr_decay, staircase=True)
my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
gradients = my_optimizer.compute_gradients(loss)
train_op = my_optimizer.apply_gradients(gradients)


sess = tf.Session()
plt.ion()

saver = tf.train.Saver()

########################################################################################################################
# LEARNING MAIN ()
########################################################################################################################

init = tf.global_variables_initializer()
sess.run(init)

if not os.path.isfile(imdb_file):
    imdb = get_imdb(learning_samples)
    with open(imdb_file, 'wb') as f:
        pickle.dump(imdb, f)
else:
    with open(imdb_file, 'rb') as f:
        imdb = pickle.load(f)

train_loss = []
validation_loss = []
for e in range(args['epochs']):

    i = 0
    sess.run(generation_num.assign(e))
    epoch_loss = 0

    while i < len(learning_samples):
        rand_idx = np.random.choice(range(len(learning_samples)), size=args['batch_size'], replace=True)
        #rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(learning_samples)])))

        rand_imgs, rand_targets = read_batch(learning_samples, learning_classes, rand_idx, imdb)

        sess.run(train_op, feed_dict={x_inputs: rand_imgs, y_targets: rand_targets})

        t_loss = sess.run(loss, feed_dict={x_inputs: rand_imgs, y_targets: rand_targets})

        #print('Learning\t Epoch \t \{\}/\{\} \t Batch \{\}/\{\} \t Loss=\{:.5f\}'.format(e + 1, args['epochs'], (i + 1) // args['batch_size'] + 1, math.ceil(len(learning_samples) / args['batch_size']), t_loss))
        i += args['batch_size']
        epoch_loss += t_loss * len(rand_idx)

    epoch_loss /= len(learning_samples)
    train_loss.append(epoch_loss)

    i = 0
    epoch_loss = 0
    while i < len(validation_samples):
        rand_idx = np.random.choice(range(len(validation_samples)), size=args['batch_size'], replace=True)
        # rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(validation_samples)])))

        rand_imgs, rand_targets = read_batch(validation_samples, validation_classes, rand_idx, imdb)

        t_loss = sess.run(loss, feed_dict={x_inputs: rand_imgs, y_targets: rand_targets})

        i += args['batch_size']
        epoch_loss += t_loss * len(rand_idx)

    epoch_loss /= len(validation_samples)
    validation_loss.append(epoch_loss)

    eval_indices = range(1, e + 2)
    fig_1 = plt.figure(1)
    plt.clf()
    plt.plot(eval_indices, train_loss, 'g-')
    plt.plot(eval_indices, validation_loss, 'r-')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')

    fig_1.show()
    plt.pause(0.01)

    ################################################################################################

    if args['patience'] > 0:
        last_difs = np.diff(np.asarray(validation_loss))
        if len(last_difs) >= args['patience']:
            if (last_difs[-args['patience']:] > 0).all():
                break

########################################################################################################################
# Save model
########################################################################################################################

saver.save(sess, os.path.join(date_time_folder, 'model'))

plt.savefig(os.path.join(date_time_folder, 'Learning.png'))

# #######################################
# Test evaluation
# #######################################


i = 0
test_out = []
test_gt = []
while i < len(test_samples):
    rand_idx = np.asarray(range(i, np.min([i + args['batch_size'], len(test_samples)])))
    rand_imgs, rand_y = read_batch(test_samples, test_classes, rand_idx, imdb)

    y_out = sess.run(model_outputs, feed_dict={x_inputs: rand_imgs})

    test_out.extend(y_out)
    test_gt.extend(rand_y)
    i += args['batch_size']


test_out = list(np.argmax(np.asarray(test_out), axis=1))
test_gt = list(np.argmax(np.asarray(test_gt), axis=1))
confusion_matrix = mt.confusion_matrix(test_gt, test_out)

accuracy = mt.accuracy_score(test_gt, test_out)

file_out = open(os.path.join(date_time_folder, 'configs.txt'), "a+")
for k in args.keys():
    file_out.write('%s: %s\n' % (k, args[k]))
file_out.close()

file_out = open(os.path.join(date_time_folder, 'results.txt'), "a+")

#print('ACCURACY: \{ :.4f \}'.format(accuracy))
file_out.write('ACCURACY: %.4f \n' % accuracy)

print('------------- CONFUSION MATRIX ----------------')
for r in confusion_matrix:
    for el in r:
        #print('\{ \}'.format(el), end='\t')
        file_out.write('%d \t'% el)
    #print('')
    file_out.write('\n')

file_out.close()

name = input('Close app?')