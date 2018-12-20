'''
	Evaluates player equities at terminal nodes of the game's public tree.
'''
import os
import numpy as np

from TerminalEquity.evaluator import evaluator
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from Game.card_combinations import card_combinations

class TerminalEquity():
	def __init__(self):
		# init call and fold matrices
		self.equity_matrix = None # [I,I] can be named as call matrix
		self.fold_matrix = None # [I,I]
		# load preflop matrix
		self._pf_equity = np.load('src/TerminalEquity/matrices/pf_equity.npy')
		# load card blocking matrix from disk if exists
		if os.path.exists('src/TerminalEquity/matrices/block_matrix.npy'):
			self._block_matrix = np.load('src/TerminalEquity/matrices/block_matrix.npy')
		else:
			self._create_block_matrix()


	def _create_block_matrix(self):
		HC, CC = constants.hand_count, constants.card_count
		self._block_matrix = np.ones([HC,HC], dtype=bool)
		for p1_card1 in range(CC):
			for p1_card2 in range(p1_card1+1, CC):
				p1_idx = card_tools.get_hole_index([p1_card1, p1_card2])
				for p2_card1 in range(CC):
					for p2_card2 in range(p2_card1+1, CC):
						p2_idx = card_tools.get_hole_index([p2_card1, p2_card2])
						if p1_card1 == p2_card1 or p1_card1 == p2_card2 or \
						   p1_card2 == p2_card1 or p1_card2 == p2_card2:
						   self._block_matrix[p1_idx, p2_idx] = 0
						   self._block_matrix[p2_idx, p1_idx] = 0


	def set_last_round_call_matrix(self, call_matrix, board_cards):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param: board_cards a non-empty vector of board cards
		@param: call_matrix a tensor where the computed matrix is stored
		'''
		HC = constants.hand_count
		if board_cards.ndim != 0:
			assert(board_cards.shape[0] == 1 or board_cards.shape[0] == 2 or board_cards.shape[0] == 5) # Only Leduc, extended Leduc, and Texas Holdem are supported
		# batch eval with only single batch, because its last round
		strength = evaluator.batch_eval_fast(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([HC,1]) # * np.ones_like(call_matrix)
		strength_view_2 = strength.reshape([1,HC]) # * np.ones_like(call_matrix)

		call_matrix[:,:]  = (strength_view_1 > strength_view_2).astype(int)
		call_matrix[:,:] -= (strength_view_1 < strength_view_2).astype(int)


	def set_inner_call_matrix(self, call_matrix, last_round_boards, street):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param last_round_boards [b,5] a non-empty vector of board cards
		@param call_matrix a tensor where the computed matrix is stored
		'''
		HC, num_boards = constants.hand_count, last_round_boards.shape[0]
		BCC, CC = constants.board_card_count, constants.card_count
		if last_round_boards.ndim != 0:
			assert(last_round_boards.shape[1] == 0 or last_round_boards.shape[1] == 2 or last_round_boards.shape[1] == 5) # Only Leduc, extended Leduc, and Texas Holdem are supported
		# evaluating all possible last round boards
		strength = evaluator.batch_eval_fast(last_round_boards) # [b,I]
		# strength from player 1 perspective for all the boards and all the card combinations
		strength_view_1 = strength.reshape([num_boards,HC,1]) # * np.ones([num_boards, HC, HC], dtype=strength.dtype)
		# strength from player 2 perspective
		strength_view_2 = strength.reshape([num_boards,1,HC]) # * np.ones_like(strength_view_1)
		#
		player_possible_mask = (strength < 0).astype(int)

		for i in range(num_boards):
			possible_mask = player_possible_mask[i].reshape([1,HC]) * player_possible_mask[i].reshape([HC,1])
			# handling hand stregths (winning probs)
			matrix_mem = (strength_view_1[i] > strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask[i]
			call_matrix[:,:] += matrix_mem

			matrix_mem = (strength_view_1[i] < strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask[i]
			call_matrix[:,:] -= matrix_mem
		# normalize sum
		num_possible_boards = card_combinations.count_last_boards_possible_boards(street)
		call_matrix[:,:] *= (1 / num_possible_boards)



	def _handle_blocking_cards(self, matrix, board):
		''' Zeroes entries in an equity matrix that correspond to invalid hands.
			 A hand is invalid if it shares any cards with the board.
		@param: matrix the matrix to modify
		@param: board a possibly empty vector of board cards
		'''
		HC, CC = constants.hand_count, constants.card_count
		possible_hand_indexes = card_tools.get_possible_hand_indexes(board)
		matrix[:,:] *= possible_hand_indexes.reshape([1,HC])
		matrix[:,:] *= possible_hand_indexes.reshape([HC,1])
		matrix[:,:] *= self._block_matrix


	def get_hand_strengths(self):
		HC = constants.hand_count
		return np.sum(self.equity_matrix, axis=0)


	def set_board(self, board):
		''' Sets the board cards for the evaluator and creates its internal data
			structures.
		@param: board a possibly empty vector of board cards
		'''
		self.board, street, HC = board, card_tools.board_to_street(board), constants.hand_count
		# set call matrix
		if street == 1:
			self.equity_matrix = self._pf_equity.copy()
		elif street == constants.streets_count:
			self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
			self.set_last_round_call_matrix(self.equity_matrix, board)
			self._handle_blocking_cards(self.equity_matrix, board)
		elif street == 2 or street == 3:
			self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
			last_round_boards = card_tools.get_last_round_boards(board)
			self.set_inner_call_matrix(self.equity_matrix, last_round_boards, street)
			self._handle_blocking_cards(self.equity_matrix, board)
		else:
			assert(False) # bad street/board
		# set fold matrix
		self.fold_matrix = np.ones([HC,HC], dtype=arguments.dtype)
		# setting cards that block each other to zero
		self._handle_blocking_cards(self.fold_matrix, board)




	# def get_call_value(self, ranges):
	# 	''' Computes (a batch of) counterfactual values that a player achieves
	# 		at a terminal node where no player has folded.
	# 	@{set_board} must be called before this function.
	# 	@param: ranges a batch of opponent ranges in an (N,K) tensor, where
	# 			N is the batch size and K is the range size
	# 	@param: result a (N,K) tensor in which to save the cfvs
	# 	'''
	# 	return np.dot(ranges, self.equity_matrix)


	# def get_fold_value(self, ranges):
	# 	''' Computes (a batch of) counterfactual values that a player achieves
	# 		at a terminal node where a player has folded.
	# 	@{set_board} must be called before this function.
	# 	@param: ranges a batch of opponent ranges in an (N,K) tensor, where
	# 			N is the batch size and K is the range size
	# 	@param: result A (N,K) tensor in which to save the cfvs. Positive cfvs
	# 			are returned, and must be negated if the player in question folded.
	# 	'''
	# 	return np.dot(ranges, self.fold_matrix)


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
