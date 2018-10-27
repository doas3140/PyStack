'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets.
'''

import tensorflow as tf


def BasicHuberLoss(delta=1.0):
    def loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, delta=delta)
    return loss




#
