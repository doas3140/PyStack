'''
	Parameters for DeepStack.
'''
import os
import numpy as np

class Parameters():
	def __init__(self):
		# whether to run on GPU
		self.gpu = False
		# the tensor datatype used for storing DeepStack's internal data
		self.dtype = np.float32
		self.int_dtype = np.int16
		# server running the ACPC dealer
		self.acpc_server = "localhost"
		# server port running the ACPC dealer
		self.acpc_server_port = 20000
		# the directory for data files
		self.data_directory = '../Data/'
		# GAME INFORMATION
		# list of pot-scaled bet sizes to use in tree
		self.bet_sizing = np.array([ [1], [1], [1] ], dtype=self.dtype)
		# the size of the game's ante, in chips
		self.ante = 100
		self.sb = 50
		self.bb = 100
		# the size of each player's stack, in chips
		self.stack = 20000
		# NEURAL NETWORK
		self.XLA = True
		# path to the neural net model
		self.model_path = './Data/Models/main/'
		# self.final_model_path = os.path.join(self.model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
		self.final_model_path = os.path.join(self.model_path, 'weights.final.hdf5')
		# path where to save tf.profiler information
		self.profiler_path = './Data/Models/PotBet/profiler'
		# the neural net architecture
		self.num_layers = 5 # (excluding output layer)
		self.num_neurons = 500
		self.learning_rate = 1e-3
		self.batch_size = 1024
		self.num_epochs = 300
		# how often to save the model during training
		self.save_epoch = 2
		# how many epochs to train for
		self.epoch_count = 10
		# TF RECORDS
		self.tfrecords_batch_size = 1024
		# DATA GENERATION
		# path to the solved poker situation data used to train the neural net
		self.data_path = 'D:/Datasets/Pystack/NoLimitTexasHoldem'
		# the number of iterations that DeepStack runs CFR for
		self.cfr_iters = 800
		# the number of preliminary CFR iterations which DeepStack doesn't
		# factor into the average strategy (included in cfr_iters)
		self.cfr_skip_iters = 500
		# how many solved poker situations are generated
		self.gen_different_boards = 48
		# how many poker situations are solved simultaneously during
		# data generation
		self.gen_batch_size = 16
		# TOTAL SITUATIONS = different_boards x batch_size
		# how many files to create (single element = ~22kB)
		self.gen_num_files = 12

		assert(self.gen_different_boards % self.gen_num_files == 0)




arguments = Parameters()
