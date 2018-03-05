# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split



#os.chdir('/home/ubuntu/modeling')


data = np.load('small_dataset.npy')
labels=np.load('labels.npy')

###here we need to add a second column to the labels dataframe which is the opposite of the first column. So if a row in column A is 1, it is 0 in column B
labels = labels.astype(np.int32, copy=False)
second_column = np.copy(labels)

second_column += -1
second_column *= -1

labels_combined=np.column_stack((labels,second_column))



data_train, data_test, labels_train, labels_test = train_test_split(data, labels_combined, test_size=0.20, random_state=42)

# Shuffle the data
data_train, labels_train = shuffle(data_train, labels_train)

data_test.shape

data_train=data_train.reshape(8000, 128, 128, 1)
data_test=data_test.reshape(2000, 128, 128, 1)
#data_train.shape






# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:


network = input_data(shape=[None, 128, 128,1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

 #Step 1: Convolution
network = conv_2d(network, 128, 1, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 256, 1, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 256, 1, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
                     
                     
                     
                     
                     # Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='test_classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(data_train, labels_train, n_epoch=100, shuffle=True, validation_set=(data_test, labels_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='test_classifier')

# Save model when training is complete to a file
model.save("test_classifier.tfl")
print("Network trained and saved as test_classifier.tfl!")

