'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets.
'''

import tensorflow as tf


# main loss
def BasicHuberLoss(delta=1.0):
	def loss(y_true, y_pred):
		return tf.losses.huber_loss(y_true, y_pred, delta=delta)
	return loss


# used only as metric
def masked_huber_loss(y_true, y_pred):
	loss = tf.losses.huber_loss(y_true, y_pred, delta=1.0)
	zero = tf.constant(0.0, dtype=tf.float32)
	batch_size = tf.cast( tf.shape(y_true)[0], dtype=tf.float32 ) # 1024
	feature_size = tf.cast( tf.shape(y_true)[1], dtype=tf.float32 ) # 1326*2
	mask = tf.where( tf.equal(y_true, zero), tf.zeros_like(y_true), tf.ones_like(y_true) )
	loss_multiplier = (batch_size * feature_size) / tf.reduce_sum(mask)
	return loss * loss_multiplier




#
