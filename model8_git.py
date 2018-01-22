# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

name = '3cv256-3cv128-fc1024-fc1024'

# define constant
DATA_PATH 		= "v10_170713_5x5_dataset_instance_norm.npz"
LABEL_PATH		= "v10_170713_5x5_label.npz"
SEED 			= 66478
IMG_SIZE 		= 5
NUM_CHANNELS 	= 74

# define hyper-parameter for train
TRAIN_EPOCH	 	 = 500
LEARNING_RATE  	 = 0.0005
DECAY_RATE 		 = 0.95
FOLD 			 = 10
TRAIN_BATCH_SIZE = 100
VALID_BATCH_SIZE = 100
DISPLAY_STEP 	 = 1
NOISE_STD 		 = 0.1

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5

# define hyper_parameter for networks
n_hidden_1 = 1024
n_hidden_2 = 1024
n_conv_1   = 256
n_conv_2   = 128

def set_weights():
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
	return W, B

def model(X, W, B, batch_size, train_phase):

    X = Gaussian_noise(X, NOISE_STD)
    
    #cv = tf.nn.bias_add(tf.nn.conv2d(X, W['c1'], strides=[1,1,1,1], padding='SAME'), B['c1'])
    cv = tf.nn.conv2d(X, W['c1'], strides=[1,1,1,1], padding='SAME')
    cv = batch_norm_old(cv, B['b1'], B['g1'])
    cv = tf.nn.relu(cv)
    
    #cv = tf.nn.bias_add(tf.nn.conv2d(cv, W['c2'], strides=[1,1,1,1], padding='SAME'), B['c2'])
    cv = tf.nn.conv2d(cv, W['c2'], strides=[1,1,1,1], padding='SAME')
    cv = batch_norm_old(cv, B['b2'], B['g2'])
    cv = tf.nn.relu(cv)
    
    shape   = cv.get_shape().as_list()
    reshape = tf.reshape(cv, [batch_size, shape[1] * shape[2] * shape[3]])
    fc      = reshape
    
    #if train: fc = tf.nn.dropout(fc, 0.5, seed=SEED)
    fc = tf.nn.elu(tf.add(tf.matmul(fc, W['h1']), B['h1']))
    #if train: fc = tf.nn.dropout(fc, 0.5, seed=SEED)
    fc = tf.nn.elu(tf.add(tf.matmul(fc, W['h2']), B['h2']))

    out  = tf.matmul(fc, W['out']) + B['out']
    return out

def Gaussian_noise(input_X, std=0.01):
    new_X = input_X + tf.random_normal(tf.shape(input_X), mean=0.0, stddev=std, dtype=tf.float32)
    return new_X

def batch_norm_new(x, phase_train):
    x = tf.layers.batch_normalization(inputs=x, axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, 
        center=True, scale=True, training=phase_train)
    return x

def batch_norm_old(x, bn_b, bn_g):
    
    beta = bn_b
    gamma = bn_g
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

    normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return normed

def batch_norm(x, bn_b, bn_g, phase_train):
    beta = bn_b
    gamma = bn_g
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed