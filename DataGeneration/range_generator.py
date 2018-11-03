'''
	Samples random probability vectors for use as player ranges.
'''
import numpy as np

from Settings.arguments import arguments
from Game.Evaluation.evaluator import evaluator
from Game.card_tools import card_tools

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
		batch_size = cards.shape[0]
		assert(mass.shape[0] == batch_size)
		card_count = cards.shape[1]
		CC, BS = card_count, batch_size
		# we terminate recursion at size of 1
		if CC == 1:
			cards[ : , 0 ] = mass.copy() # (10,1) <- (10,)
		else:
			rand = np.random.rand(batch_size)
			mass1 = mass.copy() * rand
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
		@param: range a (N,K) tensor in which to store the sampled ranges, where N is
				the number of ranges to sample and K is the range size
		@see generate_range
		'''
		batch_size = ranges.shape[0]
		BS = batch_size
		mass = np.ones([BS], dtype=arguments.dtype)
		self._generate_recursion(ranges, mass)


	def set_board(self, board):
		''' Sets the (possibly empty) board cards to sample ranges with.
			The sampled ranges will assign 0 probability to any private hands that
			share any cards with the board.
		@param: board a possibly empty vector of board cards
		'''
		hand_strengths = evaluator.batch_eval(board) # (CC,)
		possible_hand_indexes = card_tools.get_possible_hand_indexes(board) # (CC,) dtype=bool
		self.possible_hands_count = possible_hand_indexes.sum(axis=0)
		PH = self.possible_hands_count
		self.possible_hands_mask = possible_hand_indexes # .reshape([1,-1]) # (1,CC)
		non_coliding_strengths = np.zeros([PH], dtype=hand_strengths.dtype)
		non_coliding_strengths = hand_strengths[self.possible_hands_mask]
		order = np.argsort(non_coliding_strengths)
		self.reverse_order = np.argsort(order)
		self.reverse_order = self.reverse_order.reshape([1,-1]) # (1,PH) # ? - :long()
		# self.reordered_range = np.zeros([]) # ? - ar reikia?
		# self.sorted_range = np.zeros([])


	def generate_range(self, ranges):
		''' Samples a batch of random range vectors.
			Each vector is sampled indepently by randomly splitting the probability
			mass between the bottom half and the top half of the range, and then
			recursing on the two halfs.
		@{set_board} must be called first.
		@param: range a (N,K) tensor in which to store the sampled ranges, where N is
				the number of ranges to sample and K is the range size
		'''
		batch_size = ranges.shape[0]
		BS, PH = batch_size, self.possible_hands_count
		self.sorted_range = np.zeros([BS,PH], dtype=arguments.dtype)
		self._generate_sorted_range(self.sorted_range)
		# we have to reorder the the range back to undo the sort by strength
		index = self.reverse_order * np.ones(self.sorted_range.shape, dtype=arguments.int_dtype)
		self.reordered_range = np_gather(self.sorted_range, 1, index)
		mask = self.possible_hands_mask * np.ones_like(ranges, dtype=bool)
		ranges.fill(0)
		ranges[:,:] = mask * self.reordered_range




def np_gather(a, dim, index):
	''' Does gather operation: https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-gatherdim-index '''
	expanded_index = []
	for i in range(a.ndim):
		if dim==i:
			expanded_index.append( index )
		else:
			shape = [-1 if i==j else 1 for j in range(a.ndim)]
			expanded_index.append( np.arange(a.shape[i]).reshape(shape) )
	return a[expanded_index]




#
