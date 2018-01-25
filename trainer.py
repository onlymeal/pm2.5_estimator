# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

import model
import util

SEED 			= model.SEED
IMG_SIZE 		= model.IMG_SIZE
NUM_CHANNELS 	= model.NUM_CHANNELS

class network:
	def __init__(self, x, y, param):
		print("initialize trainer... "),
		self.x = x
		self.y = y

		self.W = param["weight"]
		self.B = param["bias"]
		
		self.train_epoch 		= param["train_epoch"]
		self.base_learning_rate	= param["learning_rate"]
		self.decay_rate			= param["decay_rate"]
		self.fold 				= param["fold"]
		self.train_batch_size 	= param["train_batch_size"]
		self.valid_batch_size 	= param["valid_batch_size"]
		self.display_step 		= param["display_step"]
		self.save_step	 		= param["save_step"]
		self.model				= param["model"]
		self.log_file			= param["log_file"]

		self.X 		= tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
		self.Y 		= tf.placeholder(tf.float32, [None, 1])
		self.Train 	= tf.placeholder(tf.bool, name="train_phase")
		
		self.logits   = self.model(self.X, self.W, self.B, self.train_batch_size, self.Train)
		print("Done")

	def build_graph(self, fold_idx):
		print("build graph... ")

		x_train, self.y_train, x_valid, self.y_valid = util.split_set(self.x, self.y, fold_idx+1, self.fold)

		print(x_train.shape, self.y_train.shape, x_valid.shape, self.y_valid.shape)

		x_train, x_valid = util.norm_by_std_nan(x_train, x_valid)
		
		self.x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
		self.x_valid = x_valid.reshape(x_valid.shape[0], IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

		self.num_samples_train = self.x_train.shape[0]
		self.num_samples_valid = self.x_valid.shape[0]

		batch = tf.Variable(0, dtype = tf.float32)
		
		self.learning_rate 	= tf.train.inverse_time_decay(
			self.base_learning_rate,
			batch * self.train_batch_size,
			self.num_samples_train,
			self.decay_rate
			)
		self.loss = tf.reduce_mean(tf.square(self.logits - self.Y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=batch)

		print("Done")

	def update(self):
		print("update model...")
		weight = []##################
		log_fold  = []
		for fold_idx in range(self.fold) :
			util.print_file("fold = %d / %d ---" % (fold_idx+1, self.fold), self.log_file)
			self.build_graph(fold_idx)
			with tf.Session() as sess :
				sess.run(tf.global_variables_initializer())
				log_epoch = []
				for epoch in range(self.train_epoch):
					# Train phase
					train_loss = 0.
					train_total_batch = int(self.num_samples_train / self.train_batch_size)
					for step in range(train_total_batch):
						batch_x, batch_y = util.load_batch(self.x_train, self.y_train, self.train_batch_size, step+1)
						feed_dict = {self.X : batch_x, self.Y : batch_y, self.Train : True}
						_, batch_loss, learning_rate = sess.run([self.optimizer, self.loss, self.learning_rate], feed_dict = feed_dict)
						train_loss += batch_loss
					train_loss /= train_total_batch

					# Vaidation phase
					valid_loss = 0.
					valid_total_batch = int(self.num_samples_valid / self.valid_batch_size)
					pred = []
					for step in range(valid_total_batch):
						batch_x, batch_y = util.load_batch(self.x_valid, self.y_valid, self.valid_batch_size, step+1)
						feed_dict = {self.X : batch_x, self.Y : batch_y, self.Train : False}
						batch_loss = sess.run(self.loss, feed_dict = feed_dict)
						pred.append(sess.run(self.logits, feed_dict = feed_dict))
						valid_loss += batch_loss
					valid_loss /= valid_total_batch

					# Calculate R2
					pred = np.array(pred).reshape(-1, 1)
					valid_R2 = util.get_R2(pred, self.y_valid[:pred.shape[0]])
					
					if (epoch+1) % self.display_step == 0 :
						util.print_file("Epoch %03d | valid_R2: %.4f | train_loss: %.4f | valid_loss: %.4f | learning rate: %.4f"
							%(epoch+1, valid_R2, train_loss, valid_loss, learning_rate), self.log_file)
					log_epoch.append([epoch+1, valid_R2, train_loss, valid_loss, learning_rate])
					#for i in range(pred.shape[0]):
					#	print(pred[i], self.y_valid[:pred.shape[0]][i])
					#exit()
					if (epoch+1) % self.save_step == 0 :############
						weight.append(sess.run(self.W))#################
			log_fold.append(log_epoch)
		log_fold = np.array(log_fold)
		weight = np.array(weight) ##############
		np.savez("weights", weight=weight.reshape(self.fold, int(weight.shape[0]/self.fold)))################
		util.get_result(log_fold, self.log_file)
		print("train done!!")