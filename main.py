# -*- coding:utf-8 -*-
#test
import tensorflow as tf
import numpy as np

import model
import util
import trainer

# define log file descriptor
log_file = open("log/log_"+model.name+".txt", 'w')
weight_file = "weight/"

# load dataset
data, label = util.load_data(model.DATA_PATH, model.LABEL_PATH)
print(data.shape)
print(label.shape)

weight, bias = model.set_weights()
param = {
	"model"				: model.model,
	"weight"			: weight,
	"bias"				: bias,
	"train_epoch"		: model.TRAIN_EPOCH,
	"learning_rate"    	: model.LEARNING_RATE,
	"decay_rate" 		: model.DECAY_RATE,
	"fold"             	: model.FOLD,
	"train_batch_size" 	: model.TRAIN_BATCH_SIZE,
	"valid_batch_size"  : model.VALID_BATCH_SIZE,
	"display_step"     	: model.DISPLAY_STEP,
	"save_step"			: model.SAVE_STEP,
	"log_file"			: log_file
}

trainer_network = trainer.network(data, label, param)
trainer_network.update()
log_file.close()