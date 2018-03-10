
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[2]:


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")


# In[3]:


import tensorflow as tf
#from tensorflow.contrib.slim.nets import inception
#import tensorflow.contrib.slim as slim


# In[27]:


height = 128
width =128
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

#pool3_fmaps = conv2_fmaps
pool3_fmaps = 4

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 1

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')
    print(X_reshaped.shape)

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    #pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
    print('pool3', pool3.shape)
    #pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 20 * 20])
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 128 * 128])
    print('pool3_flat', pool3_flat.shape)
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
    print('pool3_flat_drop', pool3_flat_drop.shape)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)
    print('fc1_original', fc1.shape)
    #a = fc1.shape[0]

with tf.name_scope("output"):
    ##################  ????
    fc1 = tf.reshape(fc1, shape=[50,512])
    print('fc1', fc1.shape)
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    print('logits', logits.shape)
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # tf.nn.in_top_k(predictions,targets,k,name=None) k =Number of top elements to look at for computing precision
    
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[5]:


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


# In[6]:


from sklearn.model_selection import train_test_split
from tflearn.data_utils import shuffle
data = np.load('small_dataset.npy')

data = np.array([i.astype(np.float32) for i in data])
labels = np.load('labels.npy') 
labels = np.array([i.astype(np.int32) for i in labels])

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)


# Shuffle the data
#data_train, labels_train = shuffle(data_train, labels_train)

#data_test.shape

#data_train = data_train.reshape(8000, 128, 128, 1)
#data_test = data_test.reshape(2000, 128, 128, 1)
#data_train.shape


# In[28]:


import random

#n_epochs = 1000
n_epochs = 1
batch_size = 50

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None 

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        #for iteration in range(len(data_train) // batch_size):
            #X_batch, y_batch = mnist.train.next_batch(batch_size)
        p=0
        for i in range(3):
            X_batch = data_train[i*batch_size + p : i*batch_size + batch_size + p]
            y_batch = labels_train[i*batch_size + p : i*batch_size + batch_size + p]
            p=1
            X_batch = X_batch.reshape(len(X_batch), height*width)
            y_batch = y_batch.reshape(len(y_batch))
            print("X_batch iteration", i)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            print('got past sess.run')
            n = random.randint(1,10)*batch_size
            
            data_val2 = data_train[n:n+50]
            data_val3 = data_val2.reshape(len(data_val2), height*width)
            
            labels_val2 = labels_train[n:n+50]
            labels_val3 = labels_val2.reshape(len(labels_val2))
            
            
            if i % check_interval == 0:
                #loss_val = loss.eval(feed_dict={X: mnist.validation.images,
                     #                           y: mnist.validation.labels})
                loss_val = loss.eval(feed_dict={X: data_val3, y: labels_val3})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
          
        print('logits shape', logits.shape)
        
        
        
        print('stuck here?')
        

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        #acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        
        
        print('no got to here')
        
        acc_val = accuracy.eval(feed_dict={X: data_val3, y: labels_val3})
        
        print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(epoch, acc_train * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

            
    if best_model_params:
        restore_model_params(best_model_params)
    
    acc_test = accuracy.eval(feed_dict={X: data_test, y: labels_test})
    print("Final accuracy on test set:", acc_test)
    
    save_path = saver.save(sess, "./my_model")



# In[10]:


def accuracy_(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[ ]:


# Training computation.
logits = model(tf_train_dataset)
loss = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
#valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
test_prediction = tf.nn.softmax(model(tf_test_dataset))

  feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
  _, l, predictions = sess.run([training_op, loss, train_prediction], feed_dict=feed_dict)
  if (step % 50 == 0):
    print('Minibatch loss at step %d: %f' % (step, l))
    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    #print('Validation accuracy: %.1f%%' % accuracy(
      #valid_prediction.eval(), valid_labels))
print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

