'''
	Creates TFRecords from npy files for faster training.
'''
import os
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time

from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from Game.card_tools import card_tools

class TFRecordsConverter():
	def __init__(self, batch_size):
		'''
		@param: number of batches in TFRecords file
		'''
		self.batch_size = batch_size


	def convert_npy_to_tfrecords(self, npy_dirpath, tfrecords_dirpath, start_idx):
		'''
		@param: path to npy files dir
		@param: path to destination dir (tfrecords)
		'''
		# get paths to X, Y = [b], BOARDS
		inputs, targets, boards, total_len = self._get_npy_filepaths(npy_dirpath)
		# create temp lists to store batch data
		X_temp, Y_temp = [], []
		# iterate through batches of paths
		self.counter = start_idx
		for x_path, y_path, b_path in tqdm(zip(inputs, targets, boards), total=total_len):
			# load files
			x_batch = np.load(x_path) # [batch_size x num_boards, input_size]
			y_batch = np.load(y_path) # [batch_size x num_boards, target_size]
			b_batch = np.load(b_path) # [num_boards, board_size]
			batch_size = len(x_batch) // len(b_batch)
			b_batch = self._extend_board(b_batch, batch_size) # [batch_size x num_boards, card_on_board]
			# iterate through each x,y,b
			for x, y, board in zip(x_batch, y_batch, b_batch):
				# construct nn targets and inputs
				b = card_tools.convert_board_to_nn_feature(board)
				nn_inputs = np.zeros([len(x) + len(b)], dtype=np.float32)
				nn_inputs[ :len(x) ] = x
				nn_inputs[ len(x): ] = b
				# mask targets
				ranges = x[:-1]
				mask = np.ones_like(ranges) # without pot
				mask[ ranges == 0 ] = 0
				nn_targets = y * mask
				# append one item to temp list
				X_temp.append(nn_inputs)
				Y_temp.append(nn_targets)
				# check length of temp lists
				if len(X_temp) == self.batch_size:
					self._save_tfrecord(X_temp, Y_temp, tfrecords_dirpath)
					self.counter += 1
					X_temp, Y_temp = [], []
		# save last batch (not full)
		if len(X_temp) != 0:
			self._save_tfrecord(X_temp, Y_temp, tfrecords_dirpath)


	def _extend_board(self, board_batch, batch_size):
		num_boards, num_cards = board_batch.shape
		new_boards = np.zeros([batch_size*num_boards, num_cards], dtype=board_batch.dtype)
		for i, board in enumerate(board_batch):
			new_boards[ i*batch_size:(i+1)*batch_size ] = board
		return new_boards


	def _get_npy_filepaths(self, npy_dirpath):
		filenames = [f.name for f in os.scandir(npy_dirpath)]
		# filter names
		inputs = filter(lambda x: 'inputs' in x, filenames)
		targets = filter(lambda x: 'targets' in x, filenames)
		boards = filter(lambda x: 'boards' in x, filenames)
		# sort names
		inputs =  list( sorted(inputs) 	)
		targets = list( sorted(targets) )
		boards =  list( sorted(boards) 	)
		# save total_len
		total_len = len(inputs)
		# check if len is the same
		assert(len(inputs) == len(targets) and len(targets) == len(boards))
		# check if all endings are the same
		for i in range(len(inputs)):
			# get ending (ex: 0.npy)
			inputs_file_ending = '.'.join(inputs[i].split('.')[1:])
			targets_file_ending = '.'.join(targets[i].split('.')[1:])
			boards_file_ending = '.'.join(boards[i].split('.')[1:])
			assert(inputs_file_ending == targets_file_ending)
			assert(targets_file_ending == boards_file_ending)
		# append paths
		inputs = map(lambda x: os.path.join(npy_dirpath, x), inputs)
		targets = map(lambda x: os.path.join(npy_dirpath, x), targets)
		boards = map(lambda x: os.path.join(npy_dirpath, x), boards)
		# return
		return inputs, targets, boards, total_len


	def _save_tfrecord(self, X, Y, dir_path):
		# convert lists to np arrays
		X = np.array(X)
		Y = np.array(Y)
		# shuffle X,Y
		indexes = np.arange(len(X))
		np.random.shuffle(indexes)
		X = X[indexes]
		Y = Y[indexes]
		# create file path where to save
		filename = '{}.tfrecord'.format(self.counter)
		out_path = os.path.join(dir_path, filename)
		# Open a TFRecordWriter for the output-file.
		with tf.python_io.TFRecordWriter(out_path) as writer:
			# Iterate over all the X, Y pairs.
			for x, y in zip(X, Y):
				# Convert the image to raw bytes.
				x_bytes = x.tostring()
				y_bytes = y.tostring()
				# Create a dict with the data we want to save in the
				# TFRecords file. You can add more relevant data here.
				data = {
						'input': self._wrap_bytes(x_bytes),
						'output': self._wrap_bytes(y_bytes)
					   }
				# Wrap the data as TensorFlow Features -> Example
				feature = tf.train.Features(feature=data)
				example = tf.train.Example(features=feature)
				# Serialize the data.
				serialized = example.SerializeToString()
				# Write the serialized data to the TFRecords file.
				writer.write(serialized)


	def _wrap_bytes(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _wrap_int64(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




#
