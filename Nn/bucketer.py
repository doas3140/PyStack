'''
	Assigns hands to buckets on the given board.
	For the Leduc implementation, we simply assign every possible
	set of private and board cards to a unique bucket.
'''

from Settings.game_settings import game_settings
from Game.card_tools import card_tools

class Bucketer():
	def __init__(self):
		pass


	def get_bucket_count(self):
		''' Gives the total number of buckets across all boards.
		@return the number of buckets
		'''
		CC, BC = game_settings.card_count, card_tools.get_boards_count()
		return CC * BC


	def compute_buckets(self, board):
		''' Gives a vector which maps private hands to buckets on a given board.
		@param: board a !!!non-empty vector!!! of board cards
		@return a vector which maps each private hand to a bucket index
		ex: CC = 6, board = [0,3].
			buckets = [-1 13 14 -1 16 17]
		'''
		CC = game_settings.card_count
		board_idx = card_tools.get_board_index(board) # [0,...,BC-1]
		# buckets are for each card [0,1,2,3,4,5]
		buckets = np.arange(CC, dtype=arguments.dtype)
		# shift, that changes starting point of buckets,
		# to create total of CC * BC buckets (bC)
		# can be [0, CC, 2xCC, 3xCC, (BC-1)xCC]
		shift = board_idx * CC
		# shift buckets indexes by shift scalar
		buckets += shift
		# impossible hands will have bucket number -1
		# cards that are on board are impossible ?
		for i in range(board.shape[0]):
			buckets[ board[i] ] = -1
		return buckets




#
