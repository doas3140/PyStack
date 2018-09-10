'''
	Converts between vectors over private hands and vectors over buckets.
'''

from ..Nn.bucketer import Bucketer
from ..Game.card_tools import card_tools
from ..Settings.arguments import arguments
from ..Settings.game_settings import game_settings

class BucketConversion():
	def __init__(self):
		pass

	def set_board(self, board):
		''' Sets the board cards for the bucketer.
		@param: board a non-empty vector of board cards
		'''
		pass
		# finding all strength classes


	def card_range_to_bucket_range(self, card_range, bucket_range):
		''' Converts a range vector over private hands to a range vector over buckets.
		@{set_board} must be called first. Used to create inputs to the neural net.
		@param: card_range a probability vector over private hands
		@param: bucket_range a vector in which to save the resulting probability
				vector over buckets
		'''
		pass


	def get_possible_bucket_mask(self):
		''' Gives a vector of possible buckets on the the board.
		@{set_board} must be called first.
		@return a mask vector over buckets where each entry is 1 if the bucket is
				valid, 0 if not
		'''
		pass




#
