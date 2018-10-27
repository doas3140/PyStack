'''
	Generates neural net training data by solving random poker situations.
'''
import time

from Settings.arguments import arguments
from Settings.game_settings import game_settings
from DataGeneration.random_card_generator import card_generator
from Settings.constants import constants
from DataGeneration.range_generator import RangeGenerator
from Nn.bucketer import Bucketer
from Nn.bucket_conversion import BucketConversion
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.lookahead import Lookahead
from Lookahead.resolving import Resolving
from helper_classes import Node

class DataGeneration():
	def __init__(self):
		pass

	def generate_data(self, train_data_count, valid_data_count):
		''' Generates training and validation files by sampling random poker
			situations and solving them.
			Makes two calls to @{generate_data_file}. The files are saved to
			@{arguments.data_path}, respectively appended with `valid` and `train`.
		@param: train_data_count the number of training examples to generate
		@param: valid_data_count the number of validation examples to generate
		'''
		# valid data generation
		file_name = arguments.data_path + 'valid'
		t0 = time.time()
		print('Generating validation data ...')
		self.generate_data_file(valid_data_count, file_name)
		print('valid gen time:', time.time() - t0)
		# train data generation
		file_name = arguments.data_path + 'train'
		t0 = time.time()
		print('Generating train data ...')
		self.generate_data_file(train_data_count, file_name)
		print('train gen time:', time.time() - t0)
		print('Done!')


	def generate_data_file(self, data_count, file_name):
		''' Generates data files containing examples of random poker situations with
			counterfactual values from an associated solution.
			Each poker situation is randomly generated using @{range_generator} and
			@{random_card_generator}. For description of neural net input and target
			type, see @{net_builder}.
		@param: data_count the number of examples to generate
		@param: file_name the prefix of the files where the data is saved (appended
				with `.inputs`, `.targets`, and `.mask`).
		'''
		BS, PC = arguments.gen_batch_size, constants.players_count
		BCC, CC = game_settings.board_card_count, game_settings.card_count
		range_generator = RangeGenerator()
		assert(data_count % batch_size == 0, 'data count has to be divisible by the batch size')
		batch_count = data_count / BS
		bucketer = Bucketer()
		bucket_count = bucketer.get_bucket_count()
		bC = bucket_count
		target_size = bC * PC
		targets = np.zeros([data_count, target_size], dtype=arguments.dtype)
		input_size = bC * PC + 1
		inputs = np.zeros([data_count, input_size], dtype=arguments.dtype)
		masks = np.zeros([data_count, bC], dtype=arguments.dtype) # ? - bool?
		bucket_conversion = BucketConversion()
		for b in range(batch_count):
			board = card_generator.generate_cards(BCC)
			range_generator.set_board(board)
			bucket_conversion.set_board(board)
			# generating ranges
			ranges = np.zeros([PC, BS, CC], dtype=arguments.dtype)
			for player in range(PC):
				range_generator.generate_range(ranges[player])
			# generating pot sizes between ante and stack - 0.1
			min_pot = arguments.ante
			max_pot = arguments.stack - 0.1
			pot_range = max_pot - min_pot
			random_pot_sizes = np.random.rand(BS,1) * pot_range + min_pot # (BS,1)
			# pot features are pot sizes normalized between (ante/stack,1)
			pot_size_features = random_pot_sizes.copy() * (1/arguments.stack)
			# translating ranges to features
			pot_feature_index = -1
			inputs[ b*BS:(b+1)*BS, pot_feature_index ] = pot_size_features
			player_indexes = [ (0,bC), (bC,bC*2) ]
			for player in range(PC):
				start_idx, end_idx = player_indexes[player]
				bucket_conversion.card_range_to_bucket_range(ranges[player], inputs[ b*BS:(b+1)*BS , start_idx:end_idx ])
			# computaton of values using re-solving
			values = np.zeros([PC, BS, CC], dtype=arguments.dtype)
			for i in range(BS):
				resolving = Resolving()
				current_node = Node()
				current_node.board = board
				current_node.street = 2
				current_node.current_player = constants.players.P1
				pot_size = pot_size_features[i][0] * arguments.stack
				current_node.bets = np.array([pot_size, pot_size])
				p1_range = ranges[0][i]
				p2_range = ranges[1][i]
				resolving.resolve_first_node(current_node, p1_range, p2_range)
				root_values = resolving.get_root_cfv_both_players()
				root_values *= 1 / pot_size
				values[ : , i , : ] = root_values
			# translating values to nn targets
			for player in range(PC):
				start_idx, end_idx = player_indexes[player]
				bucket_conversion.card_range_to_bucket_range(values[player], targets[ b*BS:(b+1)*BS , start_idx:end_idx ])
			# computing a mask of possible buckets
			bucket_mask = bucket_conversion.get_possible_bucket_mask()
		    masks[ b*BS:(b+1)*BS , : ] = bucket_mask * np.ones([BS,bC], dtype=bucket_mask.dtype)
		np.save(file_name + '.inputs', inputs)
		np.save(file_name + '.targets', targets)
		np.save(file_name + '.masks', masks)




data_generation = DataGeneration()
