'''
	Generates neural net training data by solving random poker situations.
'''
import os
import time
import numpy as np
from tqdm import tqdm

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from Game.card_combinations import card_combinations
from Game.card_to_string_conversion import card_to_string
from DataGeneration.range_generator import RangeGenerator
from TerminalEquity.terminal_equity import TerminalEquity
from NeuralNetwork.value_nn import ValueNn
from Lookahead.lookahead import Lookahead
from Lookahead.resolving import Resolving
from helper_classes import Node

class DataGeneration():
	def __init__(self, dirpath):
		self.dirpath = dirpath
		self.counter = 0
		# init range generator and term eq
		self.range_generator = RangeGenerator()
		self.term_eq = TerminalEquity()
		# main vars
		HC, PC = constants.hand_count, constants.players_count
		self.target_size = HC * PC
		self.input_size = HC * PC + 1


	def solve_root_node(self, board, batch_size):
		HC, PC = constants.hand_count, constants.players_count
		# set board in terminal equity and range generator
		self.term_eq.set_board(board)
		self.range_generator.set_board(self.term_eq, board)
		# init inputs and outputs
		targets = np.zeros([batch_size, self.target_size], dtype=arguments.dtype)
		inputs = np.zeros([batch_size, self.input_size], dtype=arguments.dtype)
		# generating ranges
		ranges = np.zeros([PC, batch_size, HC], dtype=arguments.dtype)
		for player in range(PC):
			self.range_generator.generate_range(ranges[player])
		# put generated ranges into inputs
		for p in range(PC):
			inputs[ : , p*HC:(p+1)*HC ] = ranges[p]
		# generating pot sizes between ante and stack - 0.1
		pot_intervals = [(100,100), (200,400), (400,2000), (2000,6000), (6000,18000)]
		# take random pot size
		random_interval = pot_intervals[ np.random.randint(len(pot_intervals)) ]
		random_pot_size = int( np.random.uniform(low=random_interval[0], high=random_interval[1]) )
		# pot features are pot sizes normalized between [ante/stack; 1]
		normalized_pot_size = random_pot_size / arguments.stack
		# put normalized pot size into inputs
		inputs[ : , -1 ].fill(normalized_pot_size)
		# set up solver
		resolving = Resolving(self.term_eq, verbose=0)
		# setting up first node
		current_node = Node()
		current_node.board = board
		current_node.street = self.street
		current_node.num_bets = 0
		current_node.current_player = constants.players.P1 if self.street == 1 else constants.players.P2
		# TODO support preflop bets
		current_node.bets = np.array([random_pot_size, random_pot_size], dtype=arguments.dtype)
		# solve this node and return cfvs of root node
		results = resolving.resolve(current_node, player_range=ranges[0], opponent_range=ranges[1])
		root_values = results.root_cfvs_both_players # [b, P, I]
		# normalize cfvs dividing by pot size
		root_values /= random_pot_size
		# put calculated cfvs into targets
		for p in range(PC):
			targets[ : , p*HC:(p+1)*HC ] = root_values[ : , p , : ]
		# return inputs [b, I x P + 1] and targets [b, I x P]
		return inputs, targets


	def solve_leaf_node(self, board, batch_size):
		HC, PC = constants.hand_count, constants.players_count
		# set board in terminal equity and range generator
		self.term_eq.set_board(board)
		self.range_generator.set_board(self.term_eq, board)
		# init inputs and outputs
		inputs = np.zeros([batch_size,self.input_size], dtype=arguments.dtype)
		targets = np.zeros([batch_size,self.target_size], dtype=arguments.dtype)
		# generating ranges
		ranges = np.zeros([PC, batch_size, HC], dtype=arguments.dtype)
		for player in range(PC):
			self.range_generator.generate_range(ranges[player])
		# put generated ranges into inputs (for each board)
		for p in range(PC):
			inputs[ : , p*HC:(p+1)*HC ] = ranges[p]
		# generating pot sizes between ante and stack - 0.1
		pot_intervals = [(100,100), (200,400), (400,2000), (2000,6000), (6000,18000)]
		# take random pot size
		random_interval = pot_intervals[ np.random.randint(len(pot_intervals)) ]
		random_pot_size = int( np.random.uniform(low=random_interval[0], high=random_interval[1]) )
		# pot features are pot sizes normalized between [ante/stack; 1]
		normalized_pot_size = random_pot_size / arguments.stack
		# put normalized pot size into inputs
		inputs[ : , -1 ].fill(normalized_pot_size)
		# set up neural network
		nn = ValueNn(self.street+1, approximate='root_nodes', pretrained_weights=True, verbose=0)
		# fill inputs into temp var for neural network to predict
		num_board_features = constants.rank_count + constants.suit_count + constants.card_count
		nn_input  = np.zeros([batch_size, self.input_size + num_board_features], dtype=arguments.dtype)
		nn_output = np.zeros([batch_size, self.target_size], dtype=arguments.dtype)
		# iterate through all possible boards
		next_boards = card_tools.get_next_round_boards(board)
		for next_board in tqdm(next_boards):
			board_features = card_tools.convert_board_to_nn_feature(next_board)
			nn_input[ : , self.input_size: ] = board_features
			mask = card_tools.get_possible_hand_indexes(next_board)
			nn_input[ : , :self.input_size ] = inputs.copy()
			nn_input[ : , 0:HC ] *= mask
			nn_input[ : , HC:2*HC ] *= mask
			nn.get_value( nn_input, nn_output )
			targets += nn_output
		# calculate targets mean (from all next boards)
		num_possible_boards = card_combinations.count_next_boards_possible_boards(self.street)
		targets *= 1 / num_possible_boards
		# return inputs [b, I x P + 1] and targets [b, I x P]
		return inputs, targets



	def generate_data(self, street, approximate='root_nodes', starting_idx=0):
		card_count = constants.card_count
		# set up scalar variables
		self.street = street
		num_board_cards = constants.board_card_count[self.street-1]
		batch_size = arguments.gen_batch_size
		num_different_boards = arguments.gen_different_boards
		total_situations = batch_size * num_different_boards
		num_files = arguments.gen_num_files
		num_batches_in_file = total_situations // num_files
		num_different_boards_per_file = num_different_boards // num_files
		for self.counter in range(starting_idx,  starting_idx + num_files):
			TARGETS = np.zeros([num_batches_in_file, self.target_size], dtype=arguments.dtype)
			INPUTS =  np.zeros([num_batches_in_file, self.input_size],  dtype=arguments.dtype)
			BOARDS = np.zeros([num_different_boards_per_file, num_board_cards], dtype=arguments.dtype)
			for b in range(num_different_boards_per_file):
				t0 = time.time()
				# create random board
				if self.street == 1: board = np.zeros([], dtype=arguments.int_dtype)
				else: board = np.random.choice(card_count, size=num_board_cards, replace=False)
				# init targets, inputs and solve it
				if approximate == 'root_nodes':
					inputs, targets = self.solve_root_node(board, batch_size)
				else: # approximate == 'leaf_nodes'
					inputs, targets = self.solve_leaf_node(board, batch_size)
				# save to placeholders for later
				TARGETS[ b*batch_size:(b+1)*batch_size , : ] = targets
				INPUTS[ b*batch_size:(b+1)*batch_size , : ] = inputs
				BOARDS[ b , : ] = board
				print('took:{}'.format(time.time()-t0))
			# save
			fpath = os.path.join(self.dirpath, '{}.{}')
			np.save(fpath.format('inputs', self.counter), INPUTS.astype(np.float32))
			np.save(fpath.format('targets', self.counter), TARGETS.astype(np.float32))
			np.save(fpath.format('boards', self.counter), BOARDS.astype(np.uint8))
			self.counter += 1




#
