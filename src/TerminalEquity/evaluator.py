'''
	Evaluates any 7 card combination
'''
import numpy as np

from Settings.constants import constants
from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from Game.card_tools import card_tools

class Evaluator():
	def __init__(self):
		self._texas_lookup = np.load('src/TerminalEquity/matrices/texas_lookup.npy')
		self._idx_to_cards = self._create_index_to_cards_matrix()


	def _create_index_to_cards_matrix(self):
		''' Returns matrix that maps hand index to players hand
		@return [I,2] :matrix that maps [hand_idx] -> [card_1, card_2]
		'''
		HC, HCC, CC = constants.hand_count, constants.hand_card_count, constants.card_count
		out = np.zeros([HC,HCC], dtype=arguments.dtype)
		for card1 in range(CC):
			for card2 in range(card1+1,CC):
				idx = card_tools.get_hand_index([card1,card2])
				out[ idx , 0 ] = card1
				out[ idx , 1 ] = card2
		return out


	def evaluate(self, hands, mask):
		''' Evaluates batches of card combinations.
			And applies mask to impossible hands
		@param: [b,2-7] :batches of hands to evaluate
		@param: [b]     :matrix to mask out impossible positions (can be all ones)
		@return [b]     :batches of evaluated hands strengths
		(2-7 depends on how many cards are on board (0-5))
		'''
		rank = self._texas_lookup[ hands[ : , 0 ] + 54 ]
		for c in range(1, hands.shape[1]):
			rank = self._texas_lookup[ hands[ : , c ] + rank + 1 ]
		rank *= mask
		rank *= -1
		return rank


	def evaluate_board(self, board):
		''' Evaluates each hand for particular board (or batches of boards)
		@param: [0-5] or [b,0-5] :board (or batches of boards)
		@return [I]   or [b,I]   :strength of all possible hands (or batches)
		'''
		HC, CC = constants.hand_count, constants.card_count
		SC, HCC = constants.suit_count, constants.hand_card_count
		if board.ndim == 2:
			boards = board
			batch_size = boards.shape[0]
			hands = np.zeros([batch_size, HC, boards.shape[1] + HCC], dtype=arguments.int_dtype)
			hands[ : , : ,  :boards.shape[1] ] = np.repeat(board.reshape([batch_size, 1, boards.shape[1]]), HC, axis=1)
			hands[ : , : , -2: ] = np.repeat(self._idx_to_cards.reshape([1, HC, HCC]), batch_size, axis=0)
			mask = np.zeros([batch_size,HC], dtype=bool)
			for i, b in enumerate(boards):
				mask[i] = card_tools.get_possible_hands_mask(b)
			hands = hands.reshape([-1, board.shape[1] + HCC])
			mask = mask.reshape([-1])
			return self.evaluate(hands, mask).reshape([batch_size, HC])
		elif board.ndim == 1:
			hands = np.zeros([HC, board.shape[0] + HCC], dtype=arguments.int_dtype)
			hands[ : ,  :board.shape[0] ] = np.repeat(board.reshape([1,board.shape[0]]), HC, axis=0)
			hands[ : , -2: ] = self._idx_to_cards.copy()
			mask = card_tools.get_possible_hands_mask(board)
			return self.evaluate(hands, mask)
		else:
			assert(False) # weird board dim




evaluator = Evaluator()
