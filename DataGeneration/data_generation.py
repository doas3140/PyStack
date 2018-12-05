'''
	Generates neural net training data by solving random poker situations.
'''
import os
import time
import numpy as np
from tqdm import tqdm

from Settings.arguments import arguments
from Settings.game_settings import game_settings
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from DataGeneration.range_generator import RangeGenerator
from TerminalEquity.terminal_equity import TerminalEquity
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
		HC, PC = game_settings.hand_count, constants.players_count
		self.target_size = HC * PC
		self.input_size = HC * PC + 1


	def solve_board(self, board, batch_size):
		HC, PC = game_settings.hand_count, constants.players_count
		# set board in terminal equity and range generator
		t0 = time.time()
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
		resolving.resolve(current_node, player_range=ranges[0], opponent_range=ranges[1])
		root_values = resolving.get_root_cfv_both_players() # [b, P, I]
		# normalize cfvs dividing by pot size
		root_values /= random_pot_size
		# put calculated cfvs into targets
		for p in range(PC):
			targets[ : , p*HC:(p+1)*HC ] = root_values[ : , p , : ]
		# return inputs [b, I x P + 1] and targets [b, I x P]
		return inputs, targets



	def generate_data(self, street, starting_idx=0):
		card_count = game_settings.card_count
		# set up scalar variables
		self.street = street
		num_board_cards = game_settings.board_card_count[self.street-1]
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
				# init targets, inputs and create random board and solve it
				board = np.random.choice(card_count, size=num_board_cards, replace=False)
				inputs, targets = self.solve_board(board, batch_size)
				# save to placeholders for later
				TARGETS[ b*batch_size:(b+1)*batch_size , : ] = targets
				INPUTS[ b*batch_size:(b+1)*batch_size , : ] = inputs
				BOARDS[ b , : ] = board
				print('{}, {}) took {} seconds'.format(self.counter, b, time.time()-t0))
			# save
			fpath = os.path.join(self.dirpath, '{}.{}')
			np.save(fpath.format('inputs', self.counter), INPUTS.astype(np.float32))
			np.save(fpath.format('targets', self.counter), TARGETS.astype(np.float32))
			np.save(fpath.format('boards', self.counter), BOARDS.astype(np.uint8))
			self.counter += 1




#
