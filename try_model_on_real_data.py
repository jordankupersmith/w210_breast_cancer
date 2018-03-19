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


data = np.load('final-dataset.npy')
labels=np.load('final-labels.npy')
#labels=np.load('labels.npy').astype(np.int)
###here we need to add a second column to the labels dataframe which is the opposite of the first column. So if a row in column A is 1, it is 0 in column B
labels = labels.astype(np.float32, copy=False)
second_column = np.copy(labels)

second_column += -1
second_column *= -1

labels_combined=np.column_stack((labels,second_column))
del labels
del second_column
data = data.astype(np.float32, copy=False)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels_combined, test_size=0.20, random_state=42)
del data
# Shuffle the data
X_train, y_train = shuffle(X_train, Y_train)

#X_test.shape

#X_train=X_train.reshape(8000, 128, 128, 1)
#X_test=X_test.reshape(2000, 128, 128, 1)


X_train=X_train.reshape(224180, 128, 128, 1)
X_test=X_test.reshape(56046, 128, 128, 1)
X_train.shape

print ("X_train", X_train.dtype)
print ("X_test", X_test.dtype)

print ("Y_train", Y_train.dtype)
print ("Y_test", Y_test.dtype)




#blabels = labels.astype(bool)
#pos_data = data[[blabels]]
#pos_labels = labels[[blabels]]
#neg_data = data[[(blabels*-1+1).astype(bool)]][:pos_data.shape[0]]
#neg_labels = np.zeros((neg_data.shape[0],))

#data = np.stack([pos_data,neg_data]).reshape(-1,128,128)
#labels = np.stack([pos_labels,neg_labels]).reshape(-1)


#labels = to_categorical(labels)

#X_train, X_test, y_train, y_test = train_test_split(
    #data, labels, test_size=0.20, random_state=42)
    
    
    
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255.0
#X_test /= 255.0


#X_train = X_train.reshape(-1, 128, 128, 1)
#X_test = X_test.reshape(-1, 128, 128, 1)




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

# Step 8: Fully-connected neural network with two outputs (0=isn't a tumor, 1=is a tumor) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
                     
                     
                     
                     
                     # Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='test_classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X_train, Y_train, n_epoch=5, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='test_classifier')

# Save model when training is complete to a file
model.save("test_classifier.tfl")
print("Network trained and saved as test_classifier.tfl!")

