# -*- coding:utf-8 -*-
import numpy as np

import model

SEED         = model.SEED
IMG_SIZE     = model.IMG_SIZE
NUM_CHANNELS = model.NUM_CHANNELS

def load_data(data_path, label_path):
	#npz가 pickle보다 압축 3배 이상 좋아 용량을 덜 차지함.
	#근데 그만큼 압축되어 있어서 메모리에 올리는 시간은 아직 비교해보지 못함
	print(":::Loding data...")
	data  = np.load(data_path)['x']
	label = np.load(label_path)['y']
	return data, label

def print_file(data, f):
    print (data)
    f.write(data+"\n")
    
    return

def get_R2(pred, y_valid):
    pred = pred.reshape(pred.shape[0], 1)
    y_valid = y_valid.reshape(y_valid.shape[0], 1)
    return 1-(np.sum(np.square(y_valid - pred)) / np.sum(np.square(y_valid - np.mean(y_valid))))

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