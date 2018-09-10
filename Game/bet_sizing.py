'''
	Gives allowed bets during a game.
	Bets are restricted to be from a list of predefined fractions of the pot.
'''

from ..Settings.arguments import arguments

class BetSizing():
    def __init__(self, pot_fractions):
        ''' Constructor
        @param: pot_fractions (num_fractions,) a list of fractions of the pot
				which are allowed as bets, sorted in ascending (min->max) order
        '''
        pass


    def get_possible_bets(self, node):
        ''' Gives the bets which are legal at a game state.
        @param: node a representation of the current game state, with fields:
        		* bets (2,): the number of chips currently committed by each player
         		* current_player (): the currently acting player
        @return (N,2) tensor where N is the number of new possible game states,
				containing N sets of new commitment levels for each player
        '''
		pass
        # compute min possible raise size
        # iterate through all bets and check if they are possible
        # take pot size after opponent bet is called
        # try all pot fractions bet and see if we can use them




#
