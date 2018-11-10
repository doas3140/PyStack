'''
	Implements the same interface as @{value_nn}, but without uses terminal
	equity evaluation instead of a neural net.

	Can be used to replace the neural net during debugging.
'''
import numpy as np

from Nn.bucketer import Bucketer
from TerminalEquity.terminal_equity import TerminalEquity
from Settings.game_settings import game_settings
from Game.card_tools import card_tools
from Settings.arguments import arguments

class MockNnTerminal():
	def __init__(self):
		''' Creates an equity matrix with entries for every possible pair of buckets.
		'''
		self.bucketer = Bucketer()
		self.bucket_count = self.bucketer.get_bucket_count()
		bC, CC = self.bucket_count, game_settings.card_count
		self.equity_matrix = np.zeros([bC,bC], dtype=arguments.dtype)
		# filling equity matrix
		boards = card_tools.get_second_round_boards()
		self.board_count = boards.shape[0]
		self.terminal_equity = TerminalEquity()
		for i in range(self.board_count):
			board = boards[i]
			self.terminal_equity.set_board(board)
			call_matrix = self.terminal_equity.get_call_matrix()
			buckets = self.bucketer.compute_buckets(board)
			for c1 in range(CC):
				for c2 in range(CC):
					b1 = buckets[c1]
					b2 = buckets[c2]
					if b1 > 0 and b2 > 0:
						matrix_entry = call_matrix[c1][c2]
						self.equity_matrix[b1,b2] = matrix_entry


	def get_value(self, inputs, outputs):
		''' Gives the expected showdown equity of the two players' ranges.
		@param: inputs An (N,I) tensor containing N instances of neural net inputs.
				See @{net_builder} for details of each input.
		@param: outputs An (N,O) tensor in which to store N sets of expected showdown
				counterfactual values for each player.
		'''
		bC = self.bucket_count
		assert(outputs.ndim == 2)
		bucket_count = outputs.shape[1] / 2
		batch_size = outputs.shape[0]
		player_indexes = [ (0, bC), (bC, 2*bC) ]
		players_count = 2
		for player in range(players_count):
			p_start, p_end = player_indexes[player] # player idx
			o_start, o_end = player_indexes[1-player] # opponent idx
			outputs[ : , p_start:p_end ] = np.dot(inputs[ : , o_start:o_end ], self.equity_matrix)




#
