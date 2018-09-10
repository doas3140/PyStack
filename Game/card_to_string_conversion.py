'''
	Converts between string and numeric representations of cards.
'''

from ..Settings.arguments import arguments
from ..Settings.game_settings import game_settings

class CardToStringConversion():
    def __init__(self):
		pass


    def card_to_suit(self, card):
        ''' Gets the suit of a card.
        @param: card () the numeric representation of the card
        @return () the index of the suit
        '''
		pass


    def card_to_rank(self, card):
        ''' Gets the rank of a card.
        @param: card () the numeric representation of the card
        @return () the index of the rank
        '''
		pass

    def card_to_string(self, card):
        ''' Converts a card's numeric representation to its string representation.
        @param: card () the numeric representation of a card
        @return the string representation of the card
        '''
        pass


    def cards_to_string(self, cards):
        pass


    def string_to_card(self, card_string):
        ''' Converts a card's string representation to its numeric representation.
        @param: card_string the string representation of a card
        @return the numeric representation of the card
        '''
        pass


    def string_to_board(self, card_string):
        ''' Converts a string representing zero or one board cards to a
			vector of numeric representations.
        @param: card_string either the empty string or a string representation
				of a card
        @return either an empty tensor or a tensor containing the numeric
				representation of the card
        '''
        pass




card_to_string = CardToStringConversion()
