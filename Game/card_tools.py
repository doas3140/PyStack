'''
	A set of tools for basic operations on cards and sets of cards.

	Several of the functions deal with "range vectors", which are probability
	vectors over the set of possible private hands. For Leduc Hold'em,
	each private hand consists of one card.
'''

from ..Settings.game_settings import game_settings
from ..Settings.arguments import arguments
from ..Settings.constants import constants

class CardTools():
    def __init__(self):
		pass


    def hand_is_possible(self, hand):
        ''' Gives whether a set of cards is valid.
        @param: hand (num_cards,): a vector of cards
        @return `true` if the tensor contains valid cards and no card is repeated
        '''
        pass


    def get_possible_hand_indexes(self, board):
        ''' Gives the private hands which are valid with a given board.
        @param: board a possibly empty vector of board cards
        @return vector (num_cards,) with an entry for every possible hand
				(private card), which is `1` if the hand shares no cards
				with the board and `0` otherwise
        '''
        pass


    def get_impossible_hand_indexes(self, board):
        ''' Gives the private hands which are invalid with a given board.
        @param: board a possibly empty vector of board cards
        @return vector (num_cards,) with an entry for every possible hand
				(private card), which is `1` if the hand shares at least
				one card with the board and `0` otherwise
        '''
        pass


    def get_uniform_range(self, board):
        ''' Gives a range vector that has uniform probability on each hand
			which is valid with a given board.
        @param: board a possibly empty vector of board cards
        @return range vector (num_cards,) where invalid hands have
				0 probability and valid hands have uniform probability
        '''
        pass


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
		pass
		# impossible_hands_probabilities.sum() will only be 0 if in range vector
		# all impossible hands are 0 probability


    def board_to_street(self, board):
        ''' Gives the current betting round based on a board vector.
        @param: board a possibly empty vector of board cards
        @return () int of the current betting round
        '''
        pass


    def get_second_round_boards(self):
        ''' Gives all possible sets of board cards for the game.
        @return (N,K) tensor, where N is the number of all possible boards,
				and K is the number of cards on each board
        '''
        pass


    def get_boards_count(self):
        ''' Gives the number of all possible boards.
        @return () int of the number of all possible boards
        '''
		pass


    def _init_board_index_table(self):
        ''' Initializes the board index table.
		@return (CC,CC) matrix, where (i,j) == (j,i), because its the
				same hand combo. matrix ex: if CC = 6:
				[[ 0,  0,  1,  2,  3,  4],
       	 		 [ 0,  0,  5,  6,  7,  8],
         		 [ 1,  5,  0,  9, 10, 11],
         		 [ 2,  6,  9,  0, 12, 13],
         		 [ 3,  7, 10, 12,  0, 14],
         		 [ 4,  8, 11, 13, 14,  0]]
        '''
		pass


    def get_board_index(self, board):
        ''' Gives a numerical index for a set of board cards.
        @param: board a non-empty vector of board cards
        @return () int of the numerical index for the board
        '''
		pass


    def normalize_range(self, board, range):
        ''' Normalizes a range vector over valid hands with a given board.
        @param: board a possibly empty vector of board cards
        @param: range (num_cards,) a range vector
        @return a modified version of `range` where each invalid hand is given 0 probability and the vector is normalized
        '''
        pass




card_tools = CardTools()
