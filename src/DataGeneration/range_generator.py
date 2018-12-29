'''
	Samples random probability vectors for use as player ranges
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from TerminalEquity.evaluator import evaluator
from Game.card_tools import card_tools

class RangeGenerator():
	def __init__(self):
		pass

	def _generate_recursion(self, cards, mass):
		''' Recursively samples a section of the range vector
		@param: [b,j] :section of the range tensor, where j is the length of the range sub-vector
		@param: [b]   :vector of remaining probability mass for each batch member
		'''
		batch_size = cards.shape[0]
		assert(mass.shape[0] == batch_size)
		card_count = cards.shape[1]
		# we terminate recursion at size of 1
		if card_count == 1:
			cards[ : , 0 ] = mass.copy() # (b,1) <- (b,)
		else:
			mass1 = mass * np.random.rand(batch_size)
			mass2 = mass - mass1
			halfSize = card_count / 2
			# if the tensor contains an odd number of cards,
			# randomize which way the middle card goes
			if halfSize % 1 != 0:
				# if end is .5 then init randomly between two numbers
				halfSize = int(halfSize - 0.5)
				halfSize = halfSize + np.random.randint(2) # (0 or 1)
			halfSize = int(halfSize)

			self._generate_recursion(cards[ : , :halfSize ], mass1)
			self._generate_recursion(cards[ : , halfSize: ], mass2)


	def _generate_sorted_range(self, ranges):
		''' Samples a batch of ranges with hands sorted by strength on the board.
		@param: [b,I] :tensor in which to store the sampled ranges
		'''
		batch_size = ranges.shape[0]
		mass = np.ones([batch_size], dtype=arguments.dtype)
		self._generate_recursion(ranges, mass)


	def set_board(self, hand_strengths, board):
		''' Sets the (possibly empty) board cards to sample ranges with.
			The sampled ranges will assign 0 probability to any private hands that
			share any cards with the board.
		@param: [I]   :strengths for all hands for following board
		@param: [0-5] :vector of board cards, where card is unique index (int)
		'''
		HC = constants.hand_count
		hand_strengths = evaluator.evaluate_board(board) if board.shape[0] == 5 else hand_strengths
		# get possible hands mask for particular board
		possible_hand_indexes = card_tools.get_possible_hands_mask(board).astype(bool)
		self.possible_hands_count = possible_hand_indexes.sum()
		self.possible_hands_mask = possible_hand_indexes.reshape([1,-1])
		# non_coliding_strengths shape: [self.possible_hands_count]
		non_coliding_strengths = hand_strengths[ possible_hand_indexes ]
		order = np.argsort(non_coliding_strengths)
		self.reverse_order = np.argsort(order)
		self.reverse_order = self.reverse_order.reshape([1,-1]) # (1,PH)


	def generate_range(self, ranges):
		''' Samples a batch of random range vectors.
			Each vector is sampled indepently by randomly splitting the probability
			mass between the bottom half and the top half of the range, and then
			recursing on the two halfs.
		@param: [b,I] :tensor in which to store the sampled ranges
		'''
		batch_size, num_possible_hands = ranges.shape[0], self.possible_hands_count
		self.sorted_range = np.zeros([batch_size, num_possible_hands], dtype=arguments.dtype)
		self._generate_sorted_range(self.sorted_range)
		# we have to reorder the range back to undo the sort by strength
		# broadcasting reverse_order: [1, possible_hands] -> [batch_size, possible_hands]
		index = np.repeat(self.reverse_order.reshape([1,-1]), batch_size, axis=0)
		self.reordered_range = np_gather(self.sorted_range, axis=1, index=index)
		# broadcasting mask: [1, possible_hands] -> [batch_size, possible_hands]
		mask = np.repeat(self.possible_hands_mask.reshape([1,-1]), batch_size, axis=0)
		ranges.fill(0)
		ranges[mask] = self.reordered_range.reshape([-1])




def np_gather(a, axis, index):
	''' Does gather operation: https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-gatherdim-index '''
	expanded_index = []
	for i in range(a.ndim):
		if axis==i:
			expanded_index.append( index )
		else:
			shape = [-1 if i==j else 1 for j in range(a.ndim)]
			expanded_index.append( np.arange(a.shape[i]).reshape(shape) )
	return a[tuple(expanded_index)]




#
