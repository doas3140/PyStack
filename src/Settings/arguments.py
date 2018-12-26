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
		# cached results path (caching only first street)
		self.cache_path = './Data/cache/'
		# self.cache_path = r'D:\Datasets\Pystack\cache'
		# GAME INFORMATION
		# list of pot-scaled bet sizes to use in tree
		self.bet_sizing = { 'preflop':[1], 'flop':[1], 'turn':[1], 'river':[1] }
		# the size of the game's ante, in chips
		self.ante = 100
		self.sb = 50
		self.bb = 100
		# the size of each player's stack, in chips
		self.stack = 20000
		# NEURAL NETWORK
		self.XLA = True
		# path to the neural net model
		# self.model_path = './Data/Models/'
		self.model_path = r'D:\Datasets\Pystack\models'
		# self.model_filename = 'weights.{epoch:02d}-{val_loss:.2f}' # show epoch and loss on filename
		self.model_filename ='weights' # without ending
		# the neural net architecture
		self.num_neurons = [500,500,500,500] # must be size of num_layers
		self.learning_rate = 1e-4
		self.batch_size = 1024
		self.num_epochs = 50
		# how often to save the model during training
		self.save_epoch = 2
		# how many epochs to train for
		self.epoch_count = 10
		# TF RECORDS
		self.tfrecords_batch_size = 1024*10 # ~200MB
		# DATA GENERATION
		# path to the solved poker situation data used to train the neural net
		# self.data_path = './Data/TrainSamples/'
		self.data_path = r'D:\Datasets\Pystack\NoLimitTexasHoldem'
		# the number of iterations that DeepStack runs CFR for
		self.cfr_iters = 800
		# the number of preliminary CFR iterations which DeepStack doesn't
		# factor into the average strategy (included in cfr_iters)
		self.cfr_skip_iters = 500
		# the number of starting iters used on approximating leaf nodes
		# after these iterations next street's root nodes are approximated and averaged
		# no need for 'river', because you get values from leaf nodes anyway (using terminal equity)
		self.leaf_nodes_iterations = {
			'preflop':780,
			'flop':500,
			'turn':500
		}
		# how many solved poker situations are generated
		self.gen_different_boards = 2
		# how many poker situations are solved simultaneously during
		# data generation
		self.gen_batch_size = 100
		# TOTAL SITUATIONS = different_boards x batch_size
		# how many files to create (single element = ~22kB)
		self.gen_num_files = 1

		assert(self.gen_different_boards % self.gen_num_files == 0)




arguments = Parameters()
