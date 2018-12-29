'''
	A set of tools for basic operations on cards and sets of cards.

	Several of the functions deal with "range vectors", which are probability
	vectors over the set of possible private hands. For Leduc Hold'em,
	each private hand consists of one card.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from Game.card_combinations import card_combinations

class CardTools():
	def __init__(self):
		pass

	def convert_board_to_nn_feature(self, board):
		'''
		@param: [0-5]     :vector of board cards, where card is unique index (int)
		@return [52+4+13] :vector of shape [total cards in deck + suit count + rank count]
		'''
		num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
		# init output
		out = np.zeros([num_cards + num_suits + num_ranks], dtype=np.float32)
		if board.ndim == 0 or board.shape[0] == 0: # no cards were placed
			return out
		assert((board >= 0).all()) # all cards are indexes 0 - 51
		# init vars
		one_hot_board = np.zeros([num_cards], dtype=np.float32)
		suit_counts = np.zeros([num_suits], dtype=np.float32)
		rank_counts = np.zeros([num_ranks], dtype=np.float32)
		# encode cards, so that all ones show what card is placed
		one_hot_board[ board ] = 1
		# count number of different suits and ranks on board
		for card in board:
			suit = card_to_string.card_to_suit(card)
			rank = card_to_string.card_to_rank(card)
			suit_counts[ suit ] += 1
			rank_counts[ rank ] += 1
		# normalize counts
		rank_counts /= num_ranks
		suit_counts /= num_suits
		# combine all arrays and return
		out[ :num_cards ] = one_hot_board
		out[ num_cards:num_cards+num_suits ] = suit_counts
		out[ num_cards+num_suits: ] = rank_counts
		return out


	def get_possible_hands_mask(self, board):
		''' Gives the private hands which are valid with a given board.
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return [I]   :vector with an entry for every possible hand (private card),
				which is `1` if the hand shares no cards with the board and `0` otherwise
		'''
		HC, CC = constants.hand_count, constants.card_count
		out = np.zeros([HC], dtype=arguments.int_dtype)
		if board.ndim == 0 or board.shape[0] == 0:
			out.fill(1)
			return out

		used = np.zeros([CC], dtype=bool)
		for card in board:
			used[ card ] = 1

		for card1 in range(CC):
			if not used[card1]:
				for card2 in range(card1+1,CC):
					if not used[card2]:
						hand = [card1, card2]
						hand_index = self.get_hand_index(hand)
						out[ hand_index ] = 1
		return out


	def same_boards(self, board1, board2):
		''' checks if board1 == board2
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@param: [0-5] :vector of board cards, where card is unique index (int)
		'''
		for card1 in board1:
			found_match = False
			for card2 in board2:
				found_match = True
				break
			if not found_match:
				return False
		return True


	def board_to_street(self, board):
		''' Gives the current betting round based on a board vector
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return int   :current betting round/street
		'''
		BCC, SC = constants.board_card_count, constants.streets_count
		if board.ndim == 0 or board.shape[0] == 0:
			return 1
		else:
			for i in range(SC):
				if board.shape[0] == BCC[i]:
					return i+1


	def _build_boards(self, boards, cur_board, out, card_index, last_index, base_index):
		CC = constants.card_count
		if card_index == last_index + 1:
			for i in range(1, last_index+1):
				boards[0][boards[1]-1][i-1] = cur_board[i-1] # (boards[0] - boards, boards[1] - index)
			out[boards[1]-1] = cur_board.copy()
			boards[1] += 1
		else:
			startindex = 1
			if card_index > base_index:
				startindex = int(cur_board[card_index-1-1] + 1)
			for i in range(startindex, CC+1):
				good = True
				for j in range(1, card_index - 1 + 1):
					if cur_board[j-1] == i:
						good = False
				if good:
					cur_board[card_index-1] = i
					self._build_boards(boards, cur_board, out, card_index+1, last_index, base_index)


	def get_next_round_boards(self, board):
		''' Gives all possible sets of board cards for the game.
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return [B,I] :tensor, where B is all possible next round boards
		'''
		BCC, CC = constants.board_card_count, constants.card_count
		street = self.board_to_street(board)
		boards_count = card_combinations.count_next_street_boards(street)
		out = np.zeros([ boards_count, BCC[street] ], dtype=arguments.int_dtype)
		boards = [out,1] # (boards, index)
		cur_board = np.zeros([ BCC[street] ], dtype=arguments.int_dtype)
		if board.ndim > 0:
			for i in range(board.shape[0]):
				cur_board[i] = board[i] + 1
		#
		self._build_boards(boards, cur_board, out, BCC[street-1] + 1, BCC[street], BCC[street-1] + 1)
		out -= 1
		return out


	def get_last_round_boards(self, board):
		''' Gives all possible sets of board cards for the game.
		@param: [0-5] :vector of board cards, where card is unique index (int)
		@return [B,I] :tensor, where B is all possible next round boards
		'''
		BCC, SC = constants.board_card_count, constants.streets_count
		street = self.board_to_street(board)
		boards_count = card_combinations.count_last_street_boards(street)
		out = np.zeros([ boards_count, BCC[SC-1] ], dtype=arguments.int_dtype)
		boards = [out,1] # (boards, index)
		cur_board = np.zeros([ BCC[SC-1] ], dtype=arguments.dtype)
		if board.ndim > 0:
			for i in range(board.shape[0]):
				cur_board[i] = board[i] + 1
		self._build_boards(boards, cur_board, out, BCC[street-1] + 1, BCC[SC-1], BCC[street-1] + 1)
		out -= 1
		return out


	def get_hand_index(self, hand):
		''' Gives a numerical index for a set of hand
		@param: [2] :vector of player private cards, where card is unique index (int)
		@return int :numerical index for the hand (0-1326)
		(first card is always smaller then second!)
		'''
		index = 1
		for i in range(len(hand)):
			index += card_combinations.choose(hand[i], i+1)
		return index - 1





card_tools = CardTools()
