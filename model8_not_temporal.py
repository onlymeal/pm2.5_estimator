# coding: utf-8

# In[ ]:


import tempfile
import urllib
import sys
import tensorflow as tf
import os
import time, datetime

import csv
import numpy as np
import pickle
import random

d = datetime.date.today()
t = time.localtime()
model_name = "model8_cv512_cv256_pool_f1024_f512(non_temporal)"
dir_name   = "log/"
pre1_name  = "[recording]"
pre2_name  = "[completed]"
time_name  = "_"+d.isoformat()+"["+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)+"]"
form_name  = ".txt"
record_name= dir_name+pre1_name+model_name+time_name+form_name
comple_name= dir_name+pre2_name+model_name+time_name+form_name
csv_name   = dir_name+"pred_"  +model_name+time_name

log = open(record_name,'w')

# In[ ]:

"""Load Data"""
print ":::Loading data...",
with open('v10_170713_5x5_dataset.pickle', 'rb') as handle:
    x_tr = pickle.load(handle)
with open('v10_170713_5x5_label.pickle', 'rb') as handle:
    y_tr = pickle.load(handle)

print "completed"
print ":::",x_tr.shape, y_tr.shape

_x_tr = np.zeros(x_tr.shape)
_y_tr = np.zeros(y_tr.shape)


# In[ ]:

def print_file(data, f):
    print data,
    f.write(data+"\n")
    
    return

def split_set(data_set, label_set, fold, k):
    """split train set and test set"""
    
    rest = int(data_set.shape[0] % k)
    quo = int(data_set.shape[0] / k)
    
    if fold != k:
        test_sect = [quo*(fold-1), quo*fold]
    else :
        test_sect = [quo*(fold-1), quo*fold+rest]
    
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []
    
    for i in range(data_set.shape[0]):
        if (test_sect[0]<=i) & (i<test_sect[1]):
            x_test.append(data_set[i])
            y_test.append(label_set[i])
        else:
            x_train.append(data_set[i])
            y_train.append(label_set[i])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_test  = np.array(y_test)
    
    return x_train, y_train, x_test, y_test

def load_batch(x_train, y_train, batch_size, n):
    
    return x_train[batch_size*(n-1):batch_size*n,], y_train[batch_size*(n-1):batch_size*n,].reshape(batch_size, 1)
    
def norm_by_std_for_y(train):
    mean = np.mean(train, 0)
    std = np.std(train, 0) + 1e-8
    
    return (train - mean)/std, mean, std
    
def norm_by_std(train, val):
    mean = np.mean(train, 0)
    std = np.std(train, 0) + 1e-8
    
    return (train - mean)/std, (val - mean)/std

def norm_by_std_nan(train, val):
    mask = np.ma.array(train, mask=np.isnan(train))
    mean = np.mean(mask, 0)
    std = np.std(mask, 0)

    train = (train - mean) / std
    train = np.where(train == np.nan, 0, train)
    train = np.nan_to_num(train)

    val = (val-mean)/std
    val = np.where(val == np.nan, 0, val)
    val = np.nan_to_num(val)
    
    return train, val

def batch_norm(x, bn_b, bn_g, scope='bn'):
    with tf.variable_scope(scope):
        beta = bn_b
        gamma = bn_g
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

        normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return normed

# In[ ]:

lr = 0.0005

fold = 10
epochs = 500
tr_batch_size = 100
ev_batch_size = 100
dis_step = 1

SEED = 66478
IMG_SIZE = 5
NUM_CHANNELS = 74

n_hidden_1 = 1024
n_hidden_2 = 256
n_conv_1   = 512
n_conv_2   = 256

W = {
    'c1' : tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, n_conv_1  ], stddev=0.1, seed=SEED, dtype=tf.float32)),
    'c2' : tf.Variable(tf.truncated_normal([3, 3, n_conv_1    , n_conv_2  ], stddev=0.1, seed=SEED, dtype=tf.float32)),
    'h1' : tf.Variable(tf.truncated_normal([IMG_SIZE * IMG_SIZE * n_conv_2  , n_hidden_1], stddev=0.1, seed=SEED, dtype=tf.float32)),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1        , n_hidden_2], stddev=0.1, seed=SEED, dtype=tf.float32)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2        , 1         ], stddev=0.1, seed=SEED, dtype=tf.float32)),
}

B = {
    'c1': tf.Variable(tf.constant(0.1, shape=[n_conv_1  ], dtype=tf.float32)),
    'b1': tf.Variable(tf.constant(0.0, shape=[n_conv_1  ], dtype=tf.float32) , name='beta' , trainable=True),
    'g1': tf.Variable(tf.constant(1.0, shape=[n_conv_1  ], dtype=tf.float32) , name='gamma', trainable=True),
    'c2': tf.Variable(tf.constant(0.1, shape=[n_conv_2  ], dtype=tf.float32)),
    'b2': tf.Variable(tf.constant(0.0, shape=[n_conv_2  ], dtype=tf.float32) , name='beta' , trainable=True),
    'g2': tf.Variable(tf.constant(1.0, shape=[n_conv_2  ], dtype=tf.float32) , name='gamma', trainable=True),
    'h1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1], dtype=tf.float32)),
    'h2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2], dtype=tf.float32)),
    'out':tf.Variable(tf.constant(0.1, shape=[1         ], dtype=tf.float32))
}  

def model(X, train=False):
    #cv1 = tf.nn.bias_add(tf.nn.conv2d(X, W['c1'], strides=[1,1,1,1], padding='SAME'), B['c1'])
    cv1 = tf.nn.conv2d(X, W['c1'], strides=[1,1,1,1], padding='SAME')
    cv1 = batch_norm(cv1, B['b1'], B['g1'])
    cv1 = tf.nn.relu(cv1)
    
    r_shape   = cv1.get_shape().as_list()
    r_reshape = tf.reshape(cv1, [r_shape[0], r_shape[1] * r_shape[2] * r_shape[3]])
    r1        = r_reshape

    #cv2 = tf.nn.bias_add(tf.nn.conv2d(cv, W['c2'], strides=[1,1,1,1], padding='SAME'), B['c2'])
    cv2 = tf.nn.conv2d(cv1, W['c2'], strides=[1,1,1,1], padding='SAME')
    cv2 = batch_norm(cv2, B['b2'], B['g2'])
    cv2 = tf.nn.relu(cv2)
    
    p2 = tf.nn.max_pool(cv2, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')

    shape   = p2.get_shape().as_list()
    reshape = tf.reshape(p2, [shape[0], shape[1] * shape[2] * shape[3]])
    fc      = reshape

    if train: fc = tf.nn.dropout(fc, 0.5, seed=SEED)
    fc = tf.nn.elu(tf.add(tf.matmul(fc, W['h1']), B['h1']))
    if train: fc = tf.nn.dropout(fc, 0.5, seed=SEED)
    fc = tf.nn.elu(tf.add(tf.matmul(fc, W['h2']), B['h2']))

    out  = tf.matmul(fc, W['out']) + B['out']
    return out


print ":::=================="
print ":::Model Validation Start"

_pred = []

for i in xrange(fold):
    print_file("fold = %d/%d----"%(i+1, fold), log)
    
    x_train, y_train, x_val, y_val = split_set(x_tr, y_tr, i+1, fold)     
    
    x_train, x_val = norm_by_std_nan(x_train, x_val)
        
    x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    x_val   = x_val.reshape(x_val.shape[0], IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    
    train_size = x_train.shape[0]

    X      = tf.placeholder(tf.float32, [tr_batch_size, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    eval_X = tf.placeholder(tf.float32, [ev_batch_size, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    Y      = tf.placeholder(tf.float32, [tr_batch_size, 1])
    
    batch = tf.Variable(0, dtype=tf.float32)
    """
    lr = tf.train.exponential_decay(0.001,                # Base learning rate.
                                    batch * batch_size,  # Current index into the dataset.
                                    x_train.shape[0],          # Decay step.
                                    0.95,                # Decay rate.
                                    staircase=True)
    """
    lr = tf.train.inverse_time_decay(0.002,
                                    batch * tr_batch_size,
                                    x_train.shape[0]*20,
                                    0.95
                                    )
    logits = model(X, train=True)
    loss   = tf.reduce_mean(tf.square(logits - Y))
    opt    = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step = batch)
    eval_p = model(eval_X)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            """train"""
            tr_loss = 0.
            total_batch = int(x_train.shape[0]/tr_batch_size)
            for k in range(total_batch):
                batch_x, batch_y = load_batch(x_train, y_train, tr_batch_size, k+1)
                _, l = sess.run([opt, loss], feed_dict={X: batch_x, Y: batch_y})
                tr_loss += l
            if (x_train.shape[0] % tr_batch_size) != 0 :
                batch_x, batch_y = load_batch(x_train, y_train, tr_batch_size, k+1, remainder = True, total_batch = total_batch)
                _, l = sess.run([opt, loss], feed_dict={X: batch_x, Y: batch_y})
            tr_loss /= (total_batch+1)

            """eval"""
            val_loss = 0.
            total_batch = int(x_val.shape[0]/ev_batch_size)
            pred = []
            for k in range(total_batch):
                batch_x, batch_y = load_batch(x_val, y_val, ev_batch_size, k+1)
                pred.append(sess.run(eval_p, feed_dict={eval_X: batch_x}))   
            pred = np.array(pred).reshape(y_val.shape[0],)
            if epoch == (epochs-1) :
                _pred.append(pred)
            val_loss = np.mean(np.square(y_val - pred))
            val_r2 = 1-(np.sum(np.square(y_val - pred)) / np.sum(np.square(y_val - np.mean(y_val))))

            if epoch % dis_step == 0:
                print_file("Epoch %03d;val_r2=%f;train_loss=%f;val_loss=%f" %(epoch, val_r2, tr_loss, val_loss), log)
                print ";lr=", sess.run(lr)

print_file(":::Model Validation Completed\n\n", log)
log.close()

_pred = np.array(_pred).reshape(73000,1)
np.savetxt(csv_name+".csv", _pred, delimiter=',')

# In[ ]:

os.rename(record_name, comple_name)
