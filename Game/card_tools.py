'''
	A set of tools for basic operations on cards and sets of cards.

	Several of the functions deal with "range vectors", which are probability
	vectors over the set of possible private hands. For Leduc Hold'em,
	each private hand consists of one card.
'''
import numpy as np

from Settings.game_settings import game_settings
from Settings.arguments import arguments
from Settings.constants import constants
from tools import tools

class CardTools():
	def __init__(self):
		self._init_board_index_table()


	def hand_is_possible(self, hand):
		''' Gives whether a set of cards is valid.
		@param: hand (num_cards,): a vector of cards
		@return `true` if the tensor contains valid cards and no card is repeated
		''' # viska galima pakeisti i np funkcija ?
		CC = game_settings.card_count
		assert (hand.min() > 0 and hand.max() <= CC, 'Illegal cards in hand' )
		used_cards = np.zeros([CC], dtype=arguments.int_dtype)
		for i in range(hand.shape[0]):
			used_cards[ hand[i] ] += 1
		return used_cards.max() < 2


	# def get_possible_hands_mask(hands):
	# 	used_cards = arguments.Tensor(hands:size(1), game_settings.card_count):fill(0)
	# 	used_cards:scatterAdd(2,hands,arguments.Tensor(hands:size(1), 7):fill(1))
	# 	local ret = torch.le(torch.max(used_cards, 2), 1):long()
	# 	if arguments.gpu then
	# 	ret = ret:cudaLong()
	# 	end
	#
	# 	return ret


	def get_possible_hand_indexes(self, board):
		''' Gives the private hands which are valid with a given board.
		@param: board a possibly empty vector of board cards
		@return vector (num_cards,) with an entry for every possible hand
				(private card), which is `1` if the hand shares no cards
				with the board and `0` otherwise
				! pakeista: 0 -> False, 1 -> True !
		'''
		CC = game_settings.card_count
		out = np.zeros([CC], dtype=bool)
		if board.ndim == 0:
			out.fill(1)
			return out
		whole_hand = np.zeros([board.shape[0] + 1], dtype=arguments.int_dtype)
		# priskiria boardus -> whole_hand, isskyrus pask. el.
		whole_hand[ :-1] = board.copy()
		for card in range(CC):
			whole_hand[-1] = card
			if self.hand_is_possible(whole_hand):
				out[card] = 1
		return out


	def get_impossible_hand_indexes(self, board):
		''' Gives the private hands which are invalid with a given board.
		@param: board a possibly empty vector of board cards
		@return vector (num_cards,) with an entry for every possible hand
				(private card), which is `1` if the hand shares at least
				one card with the board and `0` otherwise
		'''
		out = self.get_possible_hand_indexes(board)
		out = 1 - out
		return out


	def get_uniform_range(self, board):
		''' Gives a range vector that has uniform probability on each hand
			which is valid with a given board.
		@param: board a possibly empty vector of board cards
		@return range vector (num_cards,) where invalid hands have
				0 probability and valid hands have uniform probability
		'''
		out = self.get_possible_hand_indexes(board)
		out = out / out.sum()
		return out


	def get_random_range(self, board, seed=np.random.random()):
		''' Randomly samples a range vector which is valid with a given board.
		@param: board a possibly empty vector of board cards
		@param: seed () a seed for the random number generator
		@return a range vector (num_cards,) where invalid hands are given 0
				probability, each valid hand is given a probability randomly sampled
				from the uniform distribution on [0,1), and the resulting
				range is normalized
		'''
		pass


	def is_valid_range(self, range, board):
		''' Checks if a range vector is valid with a given board.
		@param: range (num_cards,) a range vector to check
		@param: board a possibly empty vector of board cards
		@return `true` if the range puts 0 probability on invalid hands and has
				total probability 1
		'''
		check = range.copy()
		only_possible_hands = (range.copy() * self.get_impossible_hand_indexes(board)).sum() == 0
		sums_to_one = abs(1.0 - range.sum()) < 0.0001
		is_valid = only_possible_hands and sums_to_one
		return is_valid


	def board_to_street(self, board):
		''' Gives the current betting round based on a board vector.
		@param: board a possibly empty vector of board cards
		@return () int of the current betting round
		'''
		if board.ndim == 0:
			return 1
		else:
			return 2


	def get_second_round_boards(self):
		''' Gives all possible sets of board cards for the game.
		@return (N,K) tensor, where N is the number of all possible boards,
				and K is the number of cards on each board
		@ex if BCC = 2:   [[0., 1.],
						   [0., 2.],
						   [0., 3.],
						   [0., 4.],
						   [0., 5.],
						   [1., 2.],
						   [1., 3.],
						   [1., 4.],
						   [1., 5.],
						   [2., 3.],
						   [2., 4.],
						   [2., 5.],
						   [3., 4.],
						   [3., 5.],
						   [4., 5.]]
		'''
		boards_count = self.get_boards_count()
		BC = boards_count
		CC = game_settings.card_count
		BCC = game_settings.board_card_count
		if BCC == 1:
			out = np.zeros([BC, 1], dtype=arguments.int_dtype)
			for card in range(CC): # TODO: -> np.arange
				out[card, 0] = card
			return out
		elif BCC == 2:
			out = np.zeros([BC, 2], dtype=arguments.int_dtype)
			board_idx = 0
			for card_1 in range(CC):
				for card_2 in range(card_1+1, CC):
					out[board_idx, 0] = card_1
					out[board_idx, 1] = card_2
					board_idx += 1
			assert (board_idx == BC, 'wrong boards count!')
			return out
		else:
			assert (False, 'unsupported board size')


	def get_boards_count(self):
		''' Gives the number of all possible boards.
		@return () int of the number of all possible boards
		'''
		CC = game_settings.card_count
		BCC = game_settings.board_card_count
		if BCC == 1:
			return CC
		elif BCC == 2:
			return (CC * (CC - 1)) / 2
		else:
			assert (False, 'unsupported board size')


	def _init_board_index_table(self):
		''' Initializes the board index table.
		@return (CC,CC) matrix, where (i,j) == (j,i), because its the
				same hand combo. matrix ex: if CC = 6:
				[[-1,  0,  1,  2,  3,  4],
		         [ 0, -1,  5,  6,  7,  8],
		         [ 1,  5, -1,  9, 10, 11],
		         [ 2,  6,  9, -1, 12, 13],
		         [ 3,  7, 10, 12, -1, 14],
		         [ 4,  8, 11, 13, 14, -1]]
		'''
		BCC, CC = game_settings.board_card_count, game_settings.card_count
		if BCC == 1:
			self._board_index_table = np.arange(CC, dtype=arguments.int_dtype) # uint?
		elif BCC == 2:
			self._board_index_table = np.full([CC,CC], -1, dtype=arguments.int_dtype) # int?
			board_idx = 0
			for card_1 in range(CC):
				for card_2 in range(card_1+1, CC):
					self._board_index_table[card_1][card_2] = board_idx
					self._board_index_table[card_2][card_1] = board_idx
					board_idx += 1
		else:
			assert(False, 'unsupported board size')


	def get_board_index(self, board):
		''' Gives a numerical index for a set of board cards.
		@param: board a non-empty vector of board cards
		@return () int of the numerical index for the board
		'''
		index = self._board_index_table # ? - nereikia copy?
		for i in range(board.shape[0]):
			index = index[ board[i] ] # ? - t[board] = tas pats?
		assert(index > 0, index)
		return index


	def get_hole_index(self, hand):
		''' Gives a numerical index for a set of hole cards.
		@param: hand a non-empty vector of hole cards, sorted
		@return the numerical index for the hand
		'''
		index = 1
		for i in range(len(hand)):
			index = index + tools.choose((hand[i]+1) - 1, i+1)
		return index - 1


	def normalize_range(self, board, range):
		''' Normalizes a range vector over valid hands with a given board.
		@param: board a possibly empty vector of board cards
		@param: range (num_cards,) a range vector
		@return a modified version of `range` where each invalid hand is given 0 probability and the vector is normalized
		'''
		pass




card_tools = CardTools()
