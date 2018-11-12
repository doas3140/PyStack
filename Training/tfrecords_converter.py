'''
	Creates TFRecords from npy files for faster training.
'''
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time

from Settings.arguments import arguments

class TFRecordsConverter():
	def __init__(self, batch_size):
		'''
		@param: number of batches in TFRecords file
		'''
		self.batch_size = batch_size


	def convert_npy_to_tfrecords(self, npy_dirpath, tfrecords_dirpath):
		'''
		@param: path to npy files dir
		@param: path to destination dir (tfrecords)
		'''
		# get paths to X, Y, MASK
		inputs, targets, masks, total_len = self._get_npy_filepaths(npy_dirpath)
		# create temp lists to store batch data
		X_temp, Y_temp, M_temp = [], [], []
		# iterate through batches of paths
		self.counter = 0
		for x_path, y_path, m_path in tqdm(zip(inputs, targets, masks), total=total_len):
			# load files
			x_batch = np.load(x_path)
			y_batch = np.load(y_path)
			m_batch = np.load(m_path)
			for x, y, m in zip(x_batch, y_batch, m_batch):
				# append one item for file to temp list
				X_temp.append(x)
				Y_temp.append(y)
				M_temp.append(m)
				# check length of temp lists
				if len(X_temp) == self.batch_size:
					self._save_tfrecord(X_temp, Y_temp, M_temp, tfrecords_dirpath)
					self.counter += 1
					X_temp, Y_temp, M_temp = [], [], []
			# save last (not full) batch
			if len(X_temp) != 0:
				self._save_tfrecord(X_temp, Y_temp, M_temp, tfrecords_dirpath)


	def _get_npy_filepaths(self, npy_dirpath):
		filenames = [f.name for f in os.scandir(npy_dirpath)]
		# filter names
		inputs = filter(lambda x: 'inputs' in x, filenames)
		targets = filter(lambda x: 'targets' in x, filenames)
		masks = filter(lambda x: 'masks' in x, filenames)
		# sort names
		inputs = sorted(inputs)
		targets = sorted(targets)
		masks = sorted(masks)
		# save total_len
		total_len = len(inputs)
		# check if len is the same
		assert(len(inputs) == len(targets) and len(targets) == len(masks))
		# check if all endings are the same
		for i in range(len(masks)):
			# get ending (ex: 0.npy)
			inputs_file_ending = '.'.join(inputs[i].split('.')[1:])
			targets_file_ending = '.'.join(targets[i].split('.')[1:])
			masks_file_ending = '.'.join(masks[i].split('.')[1:])
			assert(inputs_file_ending == targets_file_ending)
			assert(targets_file_ending == masks_file_ending)
		# append paths
		inputs = map(lambda x: os.path.join(npy_dirpath, x), inputs)
		targets = map(lambda x: os.path.join(npy_dirpath, x), targets)
		masks = map(lambda x: os.path.join(npy_dirpath, x), masks)
		# return
		return inputs, targets, masks, total_len


	def _save_tfrecord(self, X, Y, MASK, dir_path):
		filename = '{}.tfrecord'.format(self.counter)
		out_path = os.path.join(dir_path, filename)
		# Open a TFRecordWriter for the output-file.
		with tf.python_io.TFRecordWriter(out_path) as writer:
			# Iterate over all the X, Y and MASK pairs.
			for x,y,m in zip(X, Y, MASK):
				# make m twice as big [1,2] -> [1,2,1,2]
				m = np.tile(m, 2)
				# multiply by mask
				x[:-1] *= m
				y *= m
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
