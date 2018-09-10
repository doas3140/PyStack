'''
	Evaluates player equities at terminal nodes of the game's public tree.
'''

from ..Game.Evaluation.evaluator import evaluator
from ..Settings.game_settings import game_settings
from ..Settings.arguments import arguments
from ..Game.card_tools import card_tools

class TerminalEquity():
    def __init__(self):
        pass


    def get_last_round_call_matrix(self, board_cards, call_matrix):
        ''' Constructs the matrix that turns player ranges into showdown equity.
        	Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
        @param: board_cards a non-empty vector of board cards
        @param: call_matrix a tensor where the computed matrix is stored
        '''
		pass


    def _handle_blocking_cards(self, equity_matrix, board):
        ''' Zeroes entries in an equity matrix that correspond to invalid hands.
         	A hand is invalid if it shares any cards with the board.
        @param: equity_matrix the matrix to modify
        @param: board a possibly empty vector of board cards
        '''
		pass


    def _set_fold_matrix(self, board):
        ''' Sets the evaluator's fold matrix, which gives the equity for terminal
        	nodes where one player has folded.
        	Creates the matrix `B` such that for player ranges `x` and `y`, `x'By` is the equity
        	for the player who doesn't fold
        @param: board a possibly empty vector of board cards
        '''
		pass
		# setting cards that block each other to zero -
		# exactly elements on diagonal in leduc variants


    def _set_call_matrix(self, board):
        ''' Sets the evaluator's call matrix, which gives the equity for terminal
        	nodes where no player has folded.
        	For nodes in the last betting round, creates the matrix `A` such that
			for player ranges `x` and `y`, `x'Ay` is the equity for the first
			player when no player folds. For nodes in the first betting round,
			gives the weighted average of all such possible matrices.
        @param: board a possibly empty vector of board cards
        '''
		pass
        # iterate through all possible next round streets
        # averaging the values in the call matrix
        # for last round we just return the matrix


    def set_board(self, board):
        ''' Sets the board cards for the evaluator and creates its internal data
			structures.
        @param: board a possibly empty vector of board cards
        '''
        pass


    def call_value(self, ranges, result):
        ''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where no player has folded.
        @{set_board} must be called before this function.
        @param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
        @param: result a (N,K) tensor in which to save the cfvs
        '''
		pass


    def fold_value(self, ranges, result):
        ''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where a player has folded.
        @{set_board} must be called before this function.
        @param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
        @param: result A (N,K) tensor in which to save the cfvs. Positive cfvs
				are returned, and must be negated if the player in question folded.
        '''
		pass


    def get_call_matrix(self):
        ''' Returns the matrix which gives showdown equity for any ranges.
        @{set_board} must be called before this function.
        @return For nodes in the last betting round, the matrix `A` such that for
				player ranges `x` and `y`, `x'Ay` is the equity for the first
				player when no player folds. For nodes in the first betting round,
				the weighted average of all such possible matrices.
        '''
        pass


    def tree_node_call_value(self, ranges, result):
        ''' Computes the counterfactual values that both players achieve at a
			terminal node where no player has folded.
        @{set_board} must be called before this function.
        @param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
        @param: result a (2,K) tensor in which to store the cfvs for each player
        '''
        pass


    def tree_node_fold_value(self, ranges, result, folding_player):
        ''' Computes the counterfactual values that both players achieve at a
			terminal node where either player has folded.
        @{set_board} must be called before this function.
        @param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
        @param: result a (2,K) tensor in which to store the cfvs for each player
        @param: folding_player which player folded
        '''
        pass




#
