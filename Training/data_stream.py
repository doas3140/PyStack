'''
	Handles the data used for neural net training and validation.
'''

from ..Settings.arguments import arguments

class DataStream():
	def __init__(self):
		''' Reads the data from training and validation files generated with
			@{data_generation_call.generate_data}.
		'''
		pass
		# loadind valid data
		# loading train data
		# transfering data to gpu if needed


	def get_valid_batch_count(self):
		''' Gives the number of batches of validation data.
			Batch size is defined by @{arguments.train_batch_size}.
		@return the number of batches
		'''
		pass


	def get_train_batch_count(self):
		''' Gives the number of batches of training data.
			Batch size is defined by @{arguments.train_batch_size}
		@return the number of batches
		'''
		pass


	def start_epoch(self):
		''' Randomizes the order of training data.
			Done so that the data is encountered in a different order
			for each epoch.
		'''
		pass
		# data are shuffled each epoch


	def get_batch(self, inputs, targets, mask, batch_index):
		''' Returns a batch of data from a specified data set.
		@param: inputs the inputs set for the given data set
		@param: targets the targets set for the given data set
		@param: mask the masks set for the given data set
		@param: batch_index the index of the batch to return
		@return the inputs set for the batch
		@return the targets set for the batch
		@return the masks set for the batch
		'''
		pass


	def get_train_batch(self, batch_index):
		''' Returns a batch of data from the training set.
		@param: batch_index the index of the batch to return
		@return the inputs set for the batch
		@return the targets set for the batch
		@return the masks set for the batch
		'''
		pass


	def get_valid_batch(self, batch_index):
		''' Returns a batch of data from the validation set.
		@param: batch_index the index of the batch to return
		@return the inputs set for the batch
		@return the targets set for the batch
		@return the masks set for the batch
		'''
		pass




#
