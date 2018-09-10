'''
	Parameters for DeepStack.
'''

class Parameters():
    def __init__(self):
        # whether to run on GPU
        self.gpu = false
        # list of pot-scaled bet sizes to use in tree
        self.bet_sizing = [1]
        # server running the ACPC dealer
        self.acpc_server = "localhost"
        # server port running the ACPC dealer
        self.acpc_server_port = 20000
        # the number of betting rounds in the game
        self.streets_count = 2
		'''
		# the tensor datatype used for storing DeepStack's internal data
        self.Tensor = torch.FloatTensor
		'''
		# the directory for data files
        self.data_directory = '../Data/'
        # the size of the game's ante, in chips
        self.ante = 100
        # the size of each player's stack, in chips
        self.stack = 1200
        # the number of iterations that DeepStack runs CFR for
        self.cfr_iters = 1000
        # the number of preliminary CFR iterations which DeepStack doesn't
		# factor into the average strategy (included in cfr_iters)
        self.cfr_skip_iters = 500
        # how many poker situations are solved simultaneously during
		# data generation
        self.gen_batch_size = 10
        # how many poker situations are used in each neural net training batch
        self.train_batch_size = 100
        # path to the solved poker situation data used to train the neural net
        self.data_path = '../Data/TrainSamples/PotBet/'
        # path to the neural net model
        self.model_path = '../Data/Models/PotBet/'
        # the name of the neural net file
        self.value_net_name = 'final'
        # the neural net architecture
        self.net = '{nn.Linear(input_size, 50), nn.PReLU(), nn.Linear(50, output_size)}'
        # how often to save the model during training
        self.save_epoch = 2
        # how many epochs to train for
        self.epoch_count = 10
        # how many solved poker situations are generated for use as
		# training examples
        self.train_data_count = 100
        # how many solved poker situations are generated for use as
		# validation examples
        self.valid_data_count = 100
        # learning rate for neural net training
        self.learning_rate = 0.001

param = Parameters()
