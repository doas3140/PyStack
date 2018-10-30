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

from Settings.game_settings import game_settings
from Game.card_to_string_conversion import card_to_string
from Game.card_tools import card_tools
from Settings.arguments import arguments

class Evaluator():
	def __init__(self):
		pass

	def evaluate_two_card_hand(self, hand_ranks):
		''' Gives a strength representation for a hand containing two cards.
		@param: hand_ranks the rank of each card in the hand
		@return the strength value of the hand
		''' # ? - this fun
		RC = game_settings.rank_count
		# check for the pair
		hand_ranks += 1 # reikia +1 pridet pries skaiciuojant value, nes 0*x = 0
		if hand_ranks[0] == hand_ranks[1]: # hand is a pair
			hand_value = hand_ranks[0]
		else: # hand is a high card
			hand_value = hand_ranks[0] * RC + hand_ranks[1]
		return hand_value


	def evaluate_three_card_hand(self, hand_ranks):
		''' Gives a strength representation for a hand containing three cards.
		@param: hand_ranks the rank of each card in the hand
		@return the strength value of the hand
		'''
		RC = game_settings.rank_count
		hand_ranks += 1 # reikia +1 pridet pries skaiciuojant value, nes 0*x = 0
		# check for the pair
		if hand_ranks[0] == hand_ranks[1]:
			# paired hand, value of the pair goes first, value of the kicker goes second
			hand_value = hand_ranks[0] * RC + hand_ranks[2]
		elif hand_ranks[1] == hand_ranks[2]:
			# paired hand, value of the pair goes first, value of the kicker goes second
			hand_value = hand_ranks[1] * RC + hand_ranks[0]
		else:
			# hand is a high card
			hand_value = hand_ranks[0] * RC * RC + hand_ranks[1] * RC + hand_ranks[2]
		return hand_value


	def evaluate(self, hand, impossible_hand_value=None):
		''' Gives a strength representation for a two or three card hand.
		@param: hand a vector of two or three cards
		@param: [opt] impossible_hand_value the value to return
				if the hand is invalid
		@return the strength value of the hand, or `impossible_hand_value`
				if the hand is invalid
		'''
		CC = game_settings.card_count
		assert(hand.max() <= CC and hand.min() > 0, 'hand does not correspond to any cards' )
		impossible_hand_value = impossible_hand_value or -1
		if not card_tools.hand_is_possible(hand):
			return impossible_hand_value
		# we are not interested in the hand suit -
		# we will use ranks instead of cards
		hand_ranks = hand.copy()
		for i in range(hand_ranks.shape[0]):
			hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
		hand_ranks = np.sort(hand_ranks)
		if hand.shape[0] == 2:
			return self.evaluate_two_card_hand(hand_ranks)
		elif hand.shape[0] == 3:
			return self.evaluate_three_card_hand(hand_ranks)
		assert(False, 'unsupported size of hand!')


	def batch_eval(self, board, impossible_hand_value=None):
		''' Gives strength representations for all private hands
			on the given board.
		@param: board a possibly empty vector of board cards
		@param: impossible_hand_value the value to assign to hands
				which are invalid on the board
		@return a vector containing a strength value or
				`impossible_hand_value` for every private hand
		'''
		CC = game_settings.card_count
		SC = game_settings.suit_count
		hand_values = np.full([CC], -1, dtype=arguments.int_dtype)
		if board.ndim == 0:
			for hand in range(CC):
				hand_values[hand] = np.floor(hand / SC) + 1
		else:
			assert(board.shape[0] == 1 or board.shape[0] == 2, 'Incorrect board size for Leduc')
			whole_hand = np.zeros([board.shape[0] + 1], dtype=arguments.int_dtype)
			whole_hand[ :-1] = board.copy()
			for card in range(CC):
				whole_hand[-1] = card
				hand_values[card] = self.evaluate(whole_hand, impossible_hand_value)
		return hand_values





evaluator = Evaluator()
