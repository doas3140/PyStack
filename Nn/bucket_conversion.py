'''
	Converts between vectors over private hands and vectors over buckets.
'''

from Nn.bucketer import Bucketer
from Game.card_tools import card_tools
from Settings.arguments import arguments
from Settings.game_settings import game_settings

class BucketConversion():
	def __init__(self):
		self.bucketer = None
  		self.bucket_count = None
		self._range_matrix = None
		self._reverse_value_matrix = None


	def set_board(self, board):
		''' Sets the board cards for the bucketer.
		@param: board a non-empty vector of board cards
		ex: BCC = 1 => BC = 6, CC = 6
			buckets = [-1 13 14 -1 16 17]
			self._range_matrix =    	 12 13 14 15 16 17  (indexes)
										  V  V  V  V  V  V
	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
		'''
		self.bucketer = Bucketer()
  		self.bucket_count = self.bucketer.get_bucket_count() # BC*CC
		CC, bC = game_settings.card_count, self.bucket_count
		self._range_matrix = np.zeros([CC,bC], dtype=arguments.int_dtype)
 		class_ids = np.arange(bC, dtype=arguments.int_dtype)
		class_ids = class_ids.reshape([1,bC]) * np.ones([CC,bC], dtype=class_ids.int_dtype)
		buckets = self.bucketer.compute_buckets(board) # [BC*CC, ..., (BC+1)*CC)
		card_buckets = buckets.reshape([CC,1]) * np.ones([CC,bC], dtype=buckets.int_dtype)
		# finding all strength classes
		# matrix for transformation from card ranges to strength class ranges
		self._range_matrix[class_ids == card_buckets] = 1
		# matrix for transformation form class values to card values
		self._reverse_value_matrix = self._range_matrix.T.copy()


	def card_range_to_bucket_range(self, card_range, bucket_range):
		''' Converts a range vector over private hands to a range vector over buckets.
		@{set_board} must be called first. Used to create inputs to the neural net.
		@param: card_range (1, CC) a probability vector over private hands
		@param: bucket_range (1,bC) a vector in which to save the resulting probability
				vector over buckets
		'''
		bucket_range = np.dot(card_range, self._range_matrix)


	def bucket_value_to_card_value(self, bucket_value, card_value):
		''' Converts a value vector over buckets to a value vector over private hands.
		@{set_board} must be called first. Used to process neural net outputs.
		@param: bucket_value a vector of values over buckets
		@param: card_value a vector in which to save the resulting vector of values
				over private hands
		'''
		card_value = np.dot(bucket_value, self._reverse_value_matrix)


	def get_possible_bucket_mask(self):
		''' Gives a vector of possible buckets on the the board.
		@{set_board} must be called first.
		@return a mask vector over buckets where each entry is 1 if the bucket is
				valid, 0 if not
		'''
		# CC = game_settings.card_count
		# mask = np.zeros([1, self.bucket_count], dtype=arguments.dtype)
		# card_indicator = np.ones([1,CC], dtype=arguments.dtype)
		# mask = np.dot(card_indicator, self._range_matrix)
		mask = np.sum(self._range_matrix, axis=0, keepdims=True) # (1,self.bucket_count)
		return mask




#
