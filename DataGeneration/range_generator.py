'''
	Samples random probability vectors for use as player ranges.
'''

from ..Settings.arguments import arguments
from ..Game.Evaluation.evaluator import evaluator
from ..Game.card_tools import card_tools

class RangeGenerator():
	def __init__(self):
		pass

	def _generate_recursion(self, cards, mass):
		''' Recursively samples a section of the range vector.
		@param cards an (N,J) section of the range tensor, where N is the batch size
				and J is the length of the range sub-vector
		@param mass a vector of remaining probability mass for each batch member
				@see generate_range
		'''
		pass
		# we terminate recursion at size of 1
		# if the tensor contains an odd number of cards,
		# randomize which way the middle card goes


	def _generate_sorted_range(self, range):
		''' Samples a batch of ranges with hands sorted by strength on the board.
		@param: range a (N,K) tensor in which to store the sampled ranges, where N is
				the number of ranges to sample and K is the range size
		@see generate_range
		'''
		pass


	def set_board(self, board):
		''' Sets the (possibly empty) board cards to sample ranges with.
			The sampled ranges will assign 0 probability to any private hands that
			share any cards with the board.
		@param: board a possibly empty vector of board cards
		'''
		pass


	def generate_range(self, range):
		''' Samples a batch of random range vectors.
			Each vector is sampled indepently by randomly splitting the probability
			mass between the bottom half and the top half of the range, and then
			recursing on the two halfs.
		@{set_board} must be called first.
		@param: range a (N,K) tensor in which to store the sampled ranges, where N is
				the number of ranges to sample and K is the range size
		'''
		pass
		# we have to reorder the the range back to undo the sort by strength




#
