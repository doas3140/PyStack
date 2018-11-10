'''
	Converts between string and numeric representations of cards.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.game_settings import game_settings

class CardToStringConversion():
	def __init__(self):
		CC = game_settings.card_count
		self.suit_table = ['s', 'h', 'c', 'd']
		self.rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
		# card -> string table
		table = {}
		for card in range(CC):
			rank_name = self.rank_table[self.card_to_rank(card)]
			suit_name = self.suit_table[self.card_to_suit(card)]
			table[card] = rank_name + suit_name
		self.card_to_string_table = table
		# string -> card table
		table = {}
		for card in range(CC):
			table[self.card_to_string_table[card]] = card
		self.string_to_card_table = table


	def card_to_suit(self, card):
		''' Gets the suit of a card.
		@param: card () the numeric representation of the card
		@return () the index of the suit
		'''
		SC = game_settings.suit_count
		return int(card % SC)


	def card_to_rank(self, card):
		''' Gets the rank of a card.
		@param: card () the numeric representation of the card
		@return () the index of the rank
		'''
		SC = game_settings.suit_count
		return int( np.floor(card / SC) )

	def card_to_string(self, card):
		''' Converts a card's numeric representation to its string representation.
		@param: card () the numeric representation of a card
		@return the string representation of the card
		'''
		CC = game_settings.card_count
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
		CC = game_settings.card_count
		card = self.string_to_card_table[card_string]
		assert(card >= 0 and card < CC)
		return card


	def string_to_board(self, card_string):
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
			return np.array([self.string_to_card(card_string)], dtype=arguments.int_dtype)




card_to_string = CardToStringConversion()
