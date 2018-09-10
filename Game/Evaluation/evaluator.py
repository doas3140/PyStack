'''
	Evaluates hand strength in Leduc Hold'em and variants.

	Works with hands which contain two or three cards, but assumes that
	the deck contains no more than two cards of each rank
	(so three-of-a-kind is not a possible hand).

	Hand strength is given as a numerical value
	, where a lower strength means a stronger hand:
	high pair < low pair < high card < low card
'''

from ..Settings.game_settings import game_settings
from ..Game.card_to_string_conversion import card_to_string
from ..Game.card_tools import card_tools
from ..Settings.arguments import arguments

class Evaluator():
    def __init__(self):
		pass

	def evaluate_two_card_hand(self, hand_ranks):
	    ''' Gives a strength representation for a hand containing two cards.
	    @param: hand_ranks the rank of each card in the hand
	    @return the strength value of the hand
	    '''
	    pass


    def evaluate_three_card_hand(self, hand_ranks):
        ''' Gives a strength representation for a hand containing three cards.
        @param: hand_ranks the rank of each card in the hand
        @return the strength value of the hand
        '''
        pass


    def evaluate(self, hand, impossible_hand_value):
        ''' Gives a strength representation for a two or three card hand.
        @param: hand a vector of two or three cards
        @param: [opt] impossible_hand_value the value to return
				if the hand is invalid
        @return the strength value of the hand, or `impossible_hand_value`
				if the hand is invalid
        '''
		pass
        # we are not interested in the hand suit -
		# we will use ranks instead of cards


    def batch_eval(self, board, impossible_hand_value):
        ''' Gives strength representations for all private hands
			on the given board.
        @param: board a possibly empty vector of board cards
        @param: impossible_hand_value the value to assign to hands
				which are invalid on the board
        @return a vector containing a strength value or
				`impossible_hand_value` for every private hand
        '''
		pass




evaluator = Evaluator()
