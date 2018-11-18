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
		self._texas_lookup = np.load('Game/Evaluation/texas_lookup.npy')


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


	def evaluate_seven_card_hand(self, hand):
		''' Gives a strength representation for a texas hold'em hand containing seven cards.
		@param: vector of all cards in the hand
		@return the strength value of the hand
		'''
		rank = self._texas_lookup[54 + (hand[0] - 1) + 1]
		for c in range(1, hand.shape[0]):
			rank = self._texas_lookup[1 + rank + (hand[c] - 1) + 1]
		return -rank


	def evaluate(self, hand, impossible_hand_value=None):
		''' Gives a strength representation for a two or three card hand.
		@param: hand a vector of cards
		@param: [opt] impossible_hand_value the value to return
				if the hand is invalid
		@return the strength value of the hand, or `impossible_hand_value`
				if the hand is invalid
		'''
		CC = game_settings.card_count
		assert(hand.max() <= CC and hand.min() > 0) # hand does not correspond to any cards
		impossible_hand_value = impossible_hand_value or -1
		if not card_tools.hand_is_possible(hand):
			return impossible_hand_value
		# we are not interested in the hand suit -
		# we will use ranks instead of cards
		if hand.shape[0] == 2:
			hand_ranks = hand.copy()
			for i in range(hand_ranks.shape[0]):
				hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
			hand_ranks = np.sort(hand_ranks)
			return self.evaluate_two_card_hand(hand_ranks)
		elif hand.shape[0] == 3:
			hand_ranks = hand.copy()
			for i in range(hand_ranks.shape[0]):
				hand_ranks[i] = card_to_string.card_to_rank(hand_ranks[i])
			hand_ranks = np.sort(hand_ranks)
			return self.evaluate_three_card_hand(hand_ranks)
		elif hand.shape[0] == 7:
			return self.evaluate_seven_card_hand(hand)
		else:
			assert(False) # unsupported size of hand!


	def evaluate_fast(hands):
		ret = self._texas_lookup[ hands[ : , 0 ] + 54 ]
		for c in range(1, hands.shape[1]):
			ret = self._texas_lookup[ hands[ : , c ] + ret + 1 ]
		ret *= card_tools.get_possible_hands_mask(hands)
		ret *= -1
		return ret


	def batch_eval(self, board, impossible_hand_value=None):
		''' Gives strength representations for all private hands
			on the given board.
		@param: board a possibly empty vector of board cards
		@param: impossible_hand_value the value to assign to hands
				which are invalid on the board
		@return a vector containing a strength value or
				`impossible_hand_value` for every private hand
		'''
		HC, CC = game_settings.hand_count, game_settings.card_count
		SC, HCC = game_settings.suit_count, game_settings.hand_card_count

		hand_values = np.full([HC], -1, dtype=arguments.dtype)
		if board.ndim == 0: # kuhn poker
			for hand in range(CC):
				hand_values[hand] = np.floor(hand / SC)
		else:
			board_size = board.shape[0]
			assert(board_size == 1 or board_size == 2 or board_size == 5) # Incorrect board size for Leduc
			whole_hand = np.zeros([board_size + HCC], dtype=arguments.dtype)
			whole_hand[  :-HCC ] = board.copy()
			if HCC == 1:
				for card in range(CC):
					whole_hand[-1] = card
					hand_values[card] = self.evaluate(whole_hand, impossible_hand_value)
			elif HCC == 2:
				for card1 in range(CC):
					for card2 in range(card1+1, CC):
						whole_hand[-2] = card1
						whole_hand[-1] = card2
						idx = card_tools.get_hole_index( [card1,card2] )
						hand_values[idx] = self.evaluate(whole_hand, impossible_hand_value)
			else:
				assert(False) # unsupported hand_card_count
		return hand_values


	def batch_eval_fast(self, board):
		HC, CC = game_settings.hand_count, game_settings.card_count
		SC, HCC = game_settings.suit_count, game_settings.hand_card_count
		if board.ndim == 0: # kuhn poker
			return None
		elif board.ndim == 2:
			batch_size = board.shape[0]
			hands = np.zeros([batch_size, HC, board.shape[1] + HCC], dtype=arguments.dtype) # ? - long
			hands[ : , : ,  :board.shape[1] ] = board.reshape([batch_size, 1, board.shape[1]]) * np.ones([batch_size, HC, board.shape[1]], dtype=board.dtype)
			hands[ : , : , -2: ] = self._idx_to_cards.reshape([1, HC, HCC]) * np.ones([batch_size, HC, HCC], dtype=self._idx_to_cards.dtype)
			return self.evaluate_fast(hands.reshape([-1, board.shape[1] + HCC])).reshape([batch_size, HC])
		elif board.ndim == 1:
			hands = np.zeros([HC, board.shape[0] + HCC], dtype=arguments.dtype) # ? - long
			hands[ : ,  :board.shape[0] ] = board.reshape([1,board.shape[0]]) * np.ones([HC,board.shape[0]])
			hands[ : , -2: ] = self._idx_to_cards.copy()
			return self.evaluate_fast(hands)
		else:
			assert(False) # weird board dim




evaluator = Evaluator()
