'''
	Evaluates player equities at terminal nodes of the game's public tree.
'''
import numpy as np

from Game.Evaluation.evaluator import evaluator
from Settings.game_settings import game_settings
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools

class TerminalEquity():
	def __init__(self):
		self._block_matrix = np.load('TerminalEquity/block_matrix.npy')
		self._pf_equity = np.load('TerminalEquity/pf_equity.npy')
		self.equity_matrix = None # (CC,CC), can be named as call matrix
		self.fold_matrix = None # (CC,CC)
		self.matrix_mem = None #
		self.batch_size = 10


	def get_last_round_call_matrix(self, board_cards, call_matrix):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param: board_cards a non-empty vector of board cards
		@param: call_matrix a tensor where the computed matrix is stored
		'''
		HC = game_settings.hand_count
		if board_cards.ndim != 0:
			assert(board_cards.shape[0] == 1 or board_cards.shape[0] == 2 or board_cards.shape[0] == 5) # Only Leduc, extended Leduc, and Texas Holdem are supported
		strength = evaluator.batch_eval_fast(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([HC,1]) * np.ones_like(call_matrix) # ? galima broadcastint
		strength_view_2 = strength.reshape([1,HC]) * np.ones_like(call_matrix)

		call_matrix[:,:] = (strength_view_1 > strength_view_2).astype(int)
		call_matrix[:,:] -= (strength_view_1 < strength_view_2).astype(int)
		self._handle_blocking_cards(call_matrix, board_cards)


	def get_inner_call_matrix(self, board_cards, call_matrix): # TODO: can be problems
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param board_cards a non-empty vector of board cards
		@param call_matrix a tensor where the computed matrix is stored
		'''
		HC, num_boards = game_settings.hand_count, board_cards.shape[0]
		if board_cards.ndim != 0:
			assert(board_cards.shape[0] == 1 or board_cards.shape[0] == 2 or board_cards.shape[0] == 5) # Only Leduc, extended Leduc, and Texas Holdem are supported
		strength = evaluator.batch_eval_fast(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([num_boards,HC,1]) * np.ones([num_boards, HC, HC], dtype=strength.dtype) # ? galima broadcastint
		strength_view_2 = strength.reshape([num_boards,1,HC]) * np.ones_like(strength_view_1)
		possible_mask = (strength < 0).astype(call_matrix.dtype)
		for i in range(0, num_boards+1, self.batch_size): # ? - kodel +1
			i1, i2 = i, i + self.batch_size
			bs = self.batch_size
			if i + self.batch_size > num_boards:
				i1, i2 = i, num_boards
				bs = num_boards - i
			self.matrix_mem[  :bs ] = (strength_view_1[ i1:i2 ] > strength_view_2[ i1:i2 ]).copy()
			self.matrix_mem[  :bs ] *= possible_mask[ i1:i2 ].reshape([bs,1,HC]) * np.ones([bs,HC,HC], dtype=possible_mask.dtype)
			self.matrix_mem[  :bs ] *= possible_mask[ i1:i2 ].reshape([bs,HC,1]) * np.ones([bs,HC,HC], dtype=possible_mask.dtype)
			call_matrix[:,:] += np.sum(self.matrix_mem[  :bs ], axis=0) # ? - keepdims

			self.matrix_mem[  :bs ] = (strength_view_1[ i1:i2 ] < strength_view_2[ i1:i2 ]).copy()
			self.matrix_mem[  :bs ] *= possible_mask[ i1:i2 ].reshape([bs,1,HC]) * np.ones([bs,HC,HC], dtype=possible_mask.dtype)
			self.matrix_mem[  :bs ] *= possible_mask[ i1:i2 ].reshape([bs,HC,1]) * np.ones([bs,HC,HC], dtype=possible_mask.dtype)
			call_matrix[:,:] -= np.sum(self.matrix_mem[  :bs ], axis=0) # ? - keepdims

		self._handle_blocking_cards(call_matrix, board_cards)



	def _handle_blocking_cards(self, equity_matrix, board):
		''' Zeroes entries in an equity matrix that correspond to invalid hands.
			 A hand is invalid if it shares any cards with the board.
		@param: equity_matrix the matrix to modify
		@param: board a possibly empty vector of board cards
		'''
		HC, HCC, CC = game_settings.hand_count, game_settings.hand_card_count, game_settings.card_count
		possible_hand_indexes = card_tools.get_possible_hand_indexes(board) # (CC,) bool type
		equity_matrix[:,:] *= possible_hand_indexes.reshape([1,HC])
		equity_matrix[:,:] *= possible_hand_indexes.reshape([HC,1]) # np.dot can be faster
		if HCC == 2:
			equity_matrix[:,:] *= self._block_matrix
		elif HCC == 1:
			for i in range(CC):
				equity_matrix[i,i] = 0


	def _set_fold_matrix(self, board):
		''' Sets the evaluator's fold matrix, which gives the equity for terminal
			nodes where one player has folded.
			Creates the matrix `B` such that for player ranges `x` and `y`,
			`x'By` is the equity for the player who doesn't fold
		@param: board a possibly empty vector of board cards
		'''
		HC = game_settings.hand_count
		self.fold_matrix = np.ones([HC,HC], dtype=arguments.dtype)
		# setting cards that block each other to zero
		self._handle_blocking_cards(self.fold_matrix, board)


	def _set_call_matrix(self, board):
		''' Sets the evaluator's call matrix, which gives the equity for terminal
			nodes where no player has folded.
			For nodes in the last betting round, creates the matrix `A` such that
			for player ranges `x` and `y`, `x'Ay` is the equity for the first
			player when no player folds. For nodes in the first betting round,
			gives the weighted average of all such possible matrices.
		@param: board a possibly empty vector of board cards
		'''
		HC, HCC = game_settings.hand_count, game_settings.hand_card_count
		BCC, SC = game_settings.board_card_count, constants.streets_count
		CC = game_settings.card_count
		street = card_tools.board_to_street(board)
		self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
		if street == SC: # for last round we just return the matrix
			self.get_last_round_call_matrix(board, self.equity_matrix)
		elif street == 3 or street == 2: # TODO: can be problems
			# iterate through all possible next round streets
			next_round_boards = card_tools.get_last_round_boards(board)
			BC = next_round_boards.shape[0] # boards_count
			if self.matrix_mem.ndim != 3 or self.matrix_mem.shape[1] != HC or self.matrix_mem.shape[2] != HC:
				self.matrix_mem = np.zeros([self.batch_size, HC, HC], dtype=arguments.dtype)
			self.get_inner_call_matrix(next_round_boards, self.equity_matrix)
			# averaging the values in the call matrix
			cards_to_come = BCC[SC] - BCC[street]
			cards_left = CC - HCC*2 - BCC[street]
			den = tools.choose(cards_left, cards_to_come)
			self.equity_matrix *= 1/den
		elif street == 1:
			self.equity_matrix = self._pf_equity.copy()
		else:
			assert(False) # impossible street


	def get_hand_strengths(self):
		HC = game_settings.hand_count
		return np.dot(np.ones([1,HC]), self.equity_matrix)


	def set_board(self, board):
		''' Sets the board cards for the evaluator and creates its internal data
			structures.
		@param: board a possibly empty vector of board cards
		'''
		self.board = board
		self._set_call_matrix(board)
		self._set_fold_matrix(board)


	def call_value(self, ranges, result):
		''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where no player has folded.
		@{set_board} must be called before this function.
		@param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
		@param: result a (N,K) tensor in which to save the cfvs
		'''
		result[ : , : ] = np.dot(ranges, self.equity_matrix)


	def fold_value(self, ranges, result):
		''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where a player has folded.
		@{set_board} must be called before this function.
		@param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
		@param: result A (N,K) tensor in which to save the cfvs. Positive cfvs
				are returned, and must be negated if the player in question folded.
		'''
		result[ : , : ] = np.dot(ranges, self.fold_matrix)


	def get_call_matrix(self):
		''' Returns the matrix which gives showdown equity for any ranges.
		@{set_board} must be called before this function.
		@return For nodes in the last betting round, the matrix `A` such that for
				player ranges `x` and `y`, `x'Ay` is the equity for the first
				player when no player folds. For nodes in the first betting round,
				the weighted average of all such possible matrices.
		'''
		return self.equity_matrix


	def tree_node_call_value(self, ranges, result): # ? - ar reikia
		''' Computes the counterfactual values that both players achieve at a
			terminal node where no player has folded.
		@{set_board} must be called before this function.
		@param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
		@param: result a (2,K) tensor in which to store the cfvs for each player
		'''
		assert(ranges.ndim == 2 and result.ndim == 2)
		self.call_value(ranges[0].reshape([1,-1]), result[1].reshape([1,-1]))
		self.call_value(ranges[1].reshape([1,-1]), result[0].reshape([1,-1]))


	def tree_node_fold_value(self, ranges, result, folding_player): # ? - ar reikia
		''' Computes the counterfactual values that both players achieve at a
			terminal node where either player has folded.
		@{set_board} must be called before this function.
		@param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
		@param: result a (2,K) tensor in which to store the cfvs for each player
		@param: folding_player which player folded
		'''
		assert(ranges.ndim == 2 and result.ndim == 2)
		self.fold_value(ranges[0].reshape([1,-1]), result[1].reshape([1,-1])) # np?
		self.fold_value(ranges[1].reshape([1,-1]), result[0].reshape([1,-1]))
		result[folding_player] *= -1




#
