'''
	Converts between string and numeric representations of cards.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants

class CardToStringConversion():
	def __init__(self):
		CC, SC = constants.card_count, constants.suit_count
		self.suit_table = ['c', 'd', 'h', 's']
		self.rank_table = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
		# card -> rank, suit
		self.card_to_suit_table = np.zeros([CC], dtype=arguments.int_dtype)
		self.card_to_rank_table = np.zeros([CC], dtype=arguments.int_dtype)
		for card in range(CC):
			self.card_to_suit_table[card] = card % SC
			self.card_to_rank_table[card] = np.floor(card / SC)
		# card -> string table
		self.card_to_string_table = {}
		for card in range(CC):
			rank_name = self.rank_table[self.card_to_rank(card)]
			suit_name = self.suit_table[self.card_to_suit(card)]
			self.card_to_string_table[card] = rank_name + suit_name
		# string -> card table
		self.string_to_card_table = {}
		for card in range(CC):
			self.string_to_card_table[self.card_to_string_table[card]] = card


	def card_to_suit(self, card):
		''' Gets the suit of a card '''
		return self.card_to_suit_table[card]


	def card_to_rank(self, card):
		''' Gets the rank of a card '''
		return self.card_to_rank_table[card]

	def card_to_string(self, card):
		''' Converts a card's numeric representation to its string representation.
		@param: card () the numeric representation of a card
		@return the string representation of the card
		'''
		CC = constants.card_count
		assert(card >= 0 and card < CC)
		return self.card_to_string_table[card]


	def cards_to_string(self, cards):
		if cards.ndim == 0:
			return ''
		out = ''
		for card in range(cards.shape[0]):
			out += self.card_to_string(cards[card])
		return out


	def string_to_card(self, card_string):
		''' Converts a card's string representation to its numeric representation.
		@param: card_string the string representation of a card
		@return the numeric representation of the card
		'''
		CC = constants.card_count
		card = self.string_to_card_table[card_string]
		assert(card >= 0 and card < CC)
		return card


	def string_to_board(self, card_string): # verified
		''' Converts a string representing zero or one board cards to a
			vector of numeric representations.
		@param: card_string either the empty string or a string representation
				of a card
		@return either an empty tensor or a tensor containing the numeric
				representation of the card
		'''
		if card_string == '':
			return np.zeros([], dtype=arguments.int_dtype)
		else:
			num_cards = len(card_string) // 2
			board = np.zeros([num_cards], dtype=arguments.int_dtype)
			for i in range(1, num_cards+1):
				board[i-1] = self.string_to_card(card_string[ (i-1)*2:i*2 ])
			return board


	def street2name(self, street):
	    if street == 1:
	        return 'preflop'
	    elif street == 2:
	        return 'flop'
	    elif street == 3:
	        return 'turn'
	    elif street == 4:
	        return 'river'




card_to_string = CardToStringConversion()
