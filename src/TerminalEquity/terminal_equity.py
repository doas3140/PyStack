'''
	Evaluates all possbile player hands for particular board at terminal nodes
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
		self.cache = {} # maps street number -> equity matrix
		# load preflop matrix
		self.cache[1] = np.load('src/TerminalEquity/matrices/pf_equity.npy')
		# load card blocking matrix from disk if exists
		if os.path.exists('src/TerminalEquity/matrices/block_matrix.npy'):
			self._block_matrix = np.load('src/TerminalEquity/matrices/block_matrix.npy')
		else:
			self._block_matrix = self._create_block_matrix()


	def set_board(self, board):
		''' Sets the board cards for the evaluator and creates internal data structures
		@param: [0-5] :vector of board cards (int)
		'''
		self.board, street, HC = board, card_tools.board_to_street(board), constants.hand_count
		if street in self.cache:
			self.equity_matrix = self.cache[street]
		else:
			# set equity matrix
			if street == constants.streets_count:
				self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
				self._set_last_round_equity_matrix(self.equity_matrix, board)
				self._handle_blocking_cards(self.equity_matrix, board)
			elif street == 2 or street == 3:
				self.equity_matrix = np.zeros([HC,HC], dtype=arguments.dtype)
				last_round_boards = card_tools.get_last_round_boards(board)
				self._set_transitioning_equity_matrix(self.equity_matrix, last_round_boards, street)
				self._handle_blocking_cards(self.equity_matrix, board)
			else:
				assert(False) # bad street/board
			# save to cache
			self.cache[street] = self.equity_matrix.copy()
		# set fold matrix
		self.fold_matrix = np.ones([HC,HC], dtype=arguments.dtype)
		# setting cards that block each other to zero
		self._handle_blocking_cards(self.fold_matrix, board)


	def get_equity_matrix(self):
		''' Returns the matrix which gives rewards for any ranges
		@return [I,I] :for nodes in the last betting round, the matrix `A` such
				that for player ranges `x` and `y`, `x'Ay` is the equity for
				the first player when no player folds. For nodes in the first
				betting round, the weighted average of all such possible matrices
		'''
		return self.equity_matrix


	def get_fold_matrix(self):
		''' Returns the matrix which gives equity for any ranges
		@return [I,I] :matrix `B` such that for player
				ranges `x` and `y`, `x'Ay` is the equity
				for the player who doesn't fold
		'''
		return self.fold_matrix


	def get_hand_strengths(self):
		''' Get strengths of all hand combinations (I). The bigger the number is,
			the stronger the hand is for particular board
			(used in GUI app to evaluate stronger hand)
		@return [I] :strength for all hand combinations
		'''
		HC = constants.hand_count
		return np.sum(self.equity_matrix, axis=0)


	def _create_block_matrix(self):
		''' Creates boolean mask matrix for hands, that cannot be available
			if particular cards where used. (ex: if hand1 is 'KsQs', then all
			hand combinations with 'Ks' or 'Qs' should not be available)
		@return [I,I] :boolean mask for possible hands
		'''
		HC, CC = constants.hand_count, constants.card_count
		out = np.ones([HC,HC], dtype=bool)
		for p1_card1 in range(CC):
			for p1_card2 in range(p1_card1+1, CC):
				p1_idx = card_tools.get_hand_index([p1_card1, p1_card2])
				for p2_card1 in range(CC):
					for p2_card2 in range(p2_card1+1, CC):
						p2_idx = card_tools.get_hand_index([p2_card1, p2_card2])
						if p1_card1 == p2_card1 or p1_card1 == p2_card2 or \
						   p1_card2 == p2_card1 or p1_card2 == p2_card2:
						   out[p1_idx, p2_idx] = 0
						   out[p2_idx, p1_idx] = 0
		return out


	def _set_last_round_equity_matrix(self, equity_matrix, board_cards):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds
		@param: [I,I] :matrix that needs to be modified
		@param: [I,I] :board_cards a non-empty vector of board cards
		'''
		HC = constants.hand_count
		# batch eval with only single batch, because its last round
		strength = evaluator.evaluate_board(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([HC,1])
		strength_view_2 = strength.reshape([1,HC])

		equity_matrix[:,:]  = (strength_view_1 > strength_view_2).astype(int)
		equity_matrix[:,:] -= (strength_view_1 < strength_view_2).astype(int)


	def _set_transitioning_equity_matrix(self, equity_matrix, last_round_boards, street):
		''' Constructs the matrix that turns player ranges into showdown equity.
			Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
		@param: [I,I] :matrix that needs to be modified
		@param: [B,5] :all possible combinations in the last round/street
		@param: int   :current round/street
		'''
		HC, num_boards = constants.hand_count, last_round_boards.shape[0]
		BCC, CC = constants.board_card_count, constants.card_count
		# evaluating all possible last round boards
		strength = evaluator.evaluate_board(last_round_boards) # [b,I]
		# strength from player 1 perspective for all the boards and all the card combinations
		strength_view_1 = strength.reshape([num_boards,HC,1])
		# strength from player 2 perspective
		strength_view_2 = strength.reshape([num_boards,1,HC])
		#
		player_possible_mask = (strength < 0).astype(int)

		for i in range(num_boards):
			possible_mask = player_possible_mask[i].reshape([1,HC]) * player_possible_mask[i].reshape([HC,1])
			# handling hand stregths (winning probs)
			matrix_mem = (strength_view_1[i] > strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask[i]
			equity_matrix[:,:] += matrix_mem

			matrix_mem = (strength_view_1[i] < strength_view_2[i]).astype(int)
			matrix_mem *= possible_mask[i]
			equity_matrix[:,:] -= matrix_mem
		# normalize sum
		num_possible_boards = card_combinations.count_last_boards_possible_boards(street)
		equity_matrix[:,:] *= (1 / num_possible_boards)



	def _handle_blocking_cards(self, matrix, board):
		''' Zeroes entries in an equity matrix that correspond to invalid hands.
			A hand is invalid if it shares any cards with the board
		@param: [I,I] :matrix that needs to be modified
		@param: [0-5] :vector of board cards
		'''
		HC, CC = constants.hand_count, constants.card_count
		possible_hand_indexes = card_tools.get_possible_hands_mask(board)
		matrix[:,:] *= possible_hand_indexes.reshape([1,HC])
		matrix[:,:] *= possible_hand_indexes.reshape([HC,1])
		matrix[:,:] *= self._block_matrix





#
