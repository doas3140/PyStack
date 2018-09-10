'''
	Assigns hands to buckets on the given board.
	For the Leduc implementation, we simply assign every possible
	set of private and board cards to a unique bucket.
'''

from ..Settings.game_settings import game_settings
from ..Game.card_tools import card_tools

class Bucketer():
	def __init__(self):
		''' Gives the total number of buckets across all boards.
		@return the number of buckets
		'''
		pass


	def compute_buckets(self, board):
		''' Gives a vector which maps private hands to buckets on a given board.
		@param: board a non-empty vector of board cards
		@return a vector which maps each private hand to a bucket index
		'''
		pass
		# impossible hands will have bucket number -1




#
