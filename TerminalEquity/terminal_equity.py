'''
	Evaluates player equities at terminal nodes of the game's public tree.
'''

from ..Game.Evaluation.evaluator import evaluator
from ..Settings.game_settings import game_settings
from ..Settings.arguments import arguments
from ..Game.card_tools import card_tools

class TerminalEquity():
    def __init__(self):
		self.equity_matrix = None # (CC,CC), can be named as call matrix
		self.fold_matrix = None # (CC,CC)
    	self.set_board(np.zeros([]))


    def get_last_round_call_matrix(self, board_cards, call_matrix):
        ''' Constructs the matrix that turns player ranges into showdown equity.
        	Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
			is the equity for the first player when no player folds.
        @param: board_cards a non-empty vector of board cards
        @param: call_matrix a tensor where the computed matrix is stored
        '''
		CC = game_settings.card_count
		assert(board_cards.shape[0] == 1 or board_cards.shape[0] == 2, 'Only Leduc and extended Leduc are now supported' )
		strength = evaluator.batch_eval(board_cards)
		# handling hand stregths (winning probs)
		strength_view_1 = strength.reshape([CC,1]) * np.ones_like(call_matrix) # ? galima broadcastint
		strength_view_2 = strength.reshape([1,CC]) * np.ones_like(call_matrix)

		call_matrix = (strength_view_1 > strength_view_2)
		call_matrix -= (strength_view_1 < strength_view_2)
		self._handle_blocking_cards(call_matrix, board_cards)


    def _handle_blocking_cards(self, equity_matrix, board):
        ''' Zeroes entries in an equity matrix that correspond to invalid hands.
         	A hand is invalid if it shares any cards with the board.
        @param: equity_matrix the matrix to modify
        @param: board a possibly empty vector of board cards
        '''
		CC = game_settings.card_count
		possible_hand_indexes = card_tools.get_possible_hand_indexes(board) # (CC,) bool type
		equity_matrix *= possible_hand_indexes.reshape([1,CC]) * possible_hand_indexes.reshape([CC,1]) # np.dot can be faster


    def _set_fold_matrix(self, board):
        ''' Sets the evaluator's fold matrix, which gives the equity for terminal
        	nodes where one player has folded.
        	Creates the matrix `B` such that for player ranges `x` and `y`, `x'By` is the equity
        	for the player who doesn't fold
        @param: board a possibly empty vector of board cards
        '''
		CC = game_settings.card_count
		self.fold_matrix = np.ones([CC,CC], dtype=float)
		# setting cards that block each other to zero -
		# exactly elements on diagonal in leduc variants
		self.fold_matrix -= np.eye(CC).astype(self.fold_matrix)
		self._handle_blocking_cards(self.fold_matrix, board)


    def _set_call_matrix(self, board):
        ''' Sets the evaluator's call matrix, which gives the equity for terminal
        	nodes where no player has folded.
        	For nodes in the last betting round, creates the matrix `A` such that
			for player ranges `x` and `y`, `x'Ay` is the equity for the first
			player when no player folds. For nodes in the first betting round,
			gives the weighted average of all such possible matrices.
        @param: board a possibly empty vector of board cards
        '''
		CC = game_settings.card_count
		BCC = game_settings.board_card_count
		street = card_tools.board_to_street(board)
		self.equity_matrix = np.zeros([CC, CC], dtype=float) # ? - :zero()
		if street == 1:
        	# iterate through all possible next round streets
			next_round_boards = card_tools.get_second_round_boards()
			next_round_equity_matrix = np.zeros([CC, CC], dtype=float)
			for board in range(next_round_boards.shape[0]):
				next_board = next_round_boards[board]
				self.get_last_round_call_matrix(next_board, next_round_equity_matrix)
				self.equity_matrix += next_round_equity_matrix
			# averaging the values in the call matrix
			weight_constant = 1/(CC-2) if BCC == 1 else 2/((CC-2)*(CC-3)) # ?
			# tas pats: weight_constant = BCC == 1 and 1/(CC-2) or 2/((CC-2)*(CC-3))
			self.equity_matrix *= weight_constant
		elif street == 2: # for last round we just return the matrix
			self.get_last_round_call_matrix(board, self.equity_matrix)
		else:
			assert(False, 'impossible street')


    def set_board(self, board):
        ''' Sets the board cards for the evaluator and creates its internal data
			structures.
        @param: board a possibly empty vector of board cards
        '''
        self._set_call_matrix(board)
		self._set_fold_matrix(board)


    def call_value(self, ranges, result):
        ''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where no player has folded.
        @{set_board} must be called before this function.
        @param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
        @param: result a (N,K) tensor in which to save the cfvs
        '''
		result = np.dot(ranges, self.equity_matrix)


    def fold_value(self, ranges, result):
        ''' Computes (a batch of) counterfactual values that a player achieves
			at a terminal node where a player has folded.
        @{set_board} must be called before this function.
        @param: ranges a batch of opponent ranges in an (N,K) tensor, where
				N is the batch size and K is the range size
        @param: result A (N,K) tensor in which to save the cfvs. Positive cfvs
				are returned, and must be negated if the player in question folded.
        '''
		result = np.dot(ranges, self.fold_matrix)


    def get_call_matrix(self):
        ''' Returns the matrix which gives showdown equity for any ranges.
        @{set_board} must be called before this function.
        @return For nodes in the last betting round, the matrix `A` such that for
				player ranges `x` and `y`, `x'Ay` is the equity for the first
				player when no player folds. For nodes in the first betting round,
				the weighted average of all such possible matrices.
        '''
        return self.equity_matrix


    def tree_node_call_value(self, ranges, result):
        ''' Computes the counterfactual values that both players achieve at a
			terminal node where no player has folded.
        @{set_board} must be called before this function.
        @param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
        @param: result a (2,K) tensor in which to store the cfvs for each player
        '''
		assert(ranges.ndim == 2 and result.ndim == 2)
		self.call_value(ranges[0].reshape([1,-1]), result[1].reshape([1,-1]))
		self.call_value(ranges[1].reshape([1,-1]), result[0].reshape([1,-1]))


    def tree_node_fold_value(self, ranges, result, folding_player):
        ''' Computes the counterfactual values that both players achieve at a
			terminal node where either player has folded.
        @{set_board} must be called before this function.
        @param: ranges a (2,K) tensor containing ranges for each player
				(where K is the range size)
        @param: result a (2,K) tensor in which to store the cfvs for each player
        @param: folding_player which player folded
        '''
        assert(ranges.ndim == 2 and result.ndim == 2)
		self.fold_value(ranges[0].reshape([1,-1]), result[1].reshape([1,-1])) # np?
		self.fold_value(ranges[1].reshape([1,-1]), result[0].reshape([1,-1]))
		result[folding_player] *= -1




#
