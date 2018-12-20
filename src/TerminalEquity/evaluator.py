'''
	Evaluates hand strength in Leduc Hold'em and variants.

	Works with hands which contain two or three cards, but assumes that
	the deck contains no more than two cards of each rank
	(so three-of-a-kind is not a possible hand).

	Hand strength is given as a numerical value
	, where a lower strength means a stronger hand:
	high pair < low pair < high card < low card
'''
import numpy as np

from Settings.constants import constants
from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from Game.card_tools import card_tools

class Evaluator():
	def __init__(self):
		HC, HCC, CC = constants.hand_count, constants.hand_card_count, constants.card_count
		self._texas_lookup = np.load('src/TerminalEquity/matrices/texas_lookup.npy')
		self._idx_to_cards = np.zeros([HC,HCC], dtype=arguments.dtype)
		for card1 in range(CC):
			for card2 in range(card1+1,CC):
				idx = card_tools.get_hole_index([card1,card2])
				self._idx_to_cards[idx][0] = card1
				self._idx_to_cards[idx][1] = card2


	def evaluate_fast(self, hands, mask):
		rank = self._texas_lookup[ hands[ : , 0 ] + 54 ]
		for c in range(1, hands.shape[1]):
			rank = self._texas_lookup[ hands[ : , c ] + rank + 1 ]
		rank *= mask
		rank *= -1
		return rank


	def batch_eval_fast(self, board): # works the same as previous one (faster 40ms -> 3)
		HC, CC = constants.hand_count, constants.card_count
		SC, HCC = constants.suit_count, constants.hand_card_count
		if board.ndim == 0: # kuhn poker
			return None
		elif board.ndim == 2:
			boards = board
			batch_size = boards.shape[0]
			hands = np.zeros([batch_size, HC, boards.shape[1] + HCC], dtype=arguments.int_dtype)
			hands[ : , : ,  :boards.shape[1] ] = board.reshape([batch_size, 1, boards.shape[1]]) * np.ones([batch_size, HC, boards.shape[1]], dtype=board.dtype)
			hands[ : , : , -2: ] = self._idx_to_cards.reshape([1, HC, HCC]) * np.ones([batch_size, HC, HCC], dtype=self._idx_to_cards.dtype)
			mask = np.zeros([batch_size,HC], dtype=bool)
			for i, b in enumerate(boards):
				mask[i] = card_tools.get_possible_hand_indexes(b)
			hands = hands.reshape([-1, board.shape[1] + HCC])
			mask = mask.reshape([-1])
			return self.evaluate_fast(hands, mask).reshape([batch_size, HC])
		elif board.ndim == 1:
			hands = np.zeros([HC, board.shape[0] + HCC], dtype=arguments.int_dtype)
			hands[ : ,  :board.shape[0] ] = board.reshape([1,board.shape[0]]) * np.ones([HC,board.shape[0]])
			hands[ : , -2: ] = self._idx_to_cards.copy()
			mask = card_tools.get_possible_hand_indexes(board)
			return self.evaluate_fast(hands, mask)
		else:
			assert(False) # weird board dim




evaluator = Evaluator()
