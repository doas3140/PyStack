'''
	Uses the neural net to estimate value at the end of the first betting round.
'''

from ..Nn.bucketer import Bucketer
from ..Game.card_tools import card_tools
from ..Settings.arguments import arguments
from ..Settings.game_settings import game_settings
from ..Settings.constants import constants

class NextRoundValue():
	def __init__(self, nn):
		''' Creates a tensor that can translate hand ranges to bucket ranges
			on any board.
		@param: nn the neural network
		'''
		self.nn = nn
    	self._init_bucketing()


	def _init_bucketing(self):
		''' Initializes the tensor that translates hand ranges to bucket ranges.
		'''
		pass
		# finding all strength classes
        # matrix for transformation from card ranges to strength class ranges
		# matrix for transformation from class values to card values
		# we need to div the matrix by the sum of possible boards
		# (from point of view of each hand)


	def _card_range_to_bucket_range(self, card_range, bucket_range):
		''' Converts a range vector over private hands to a range vector over buckets.
		@param: card_range a probability vector over private hands
		@param: bucket_range a vector in which to store the output probabilities
				over buckets
		'''
		pass


	def _bucket_value_to_card_value(self, bucket_value, card_value):
		''' Converts a value vector over buckets to a value vector over private hands.
		@param: bucket_value a value vector over buckets
		@param: card_value a vector in which to store the output values over
				private hands
		'''
		pass


	def _bucket_value_to_card_value_on_board(self, board, bucket_value, card_value):
		''' Converts a value vector over buckets to a value vector over
			private hands given a particular set of board cards.
		@param: board a non-empty vector of board cards
		@param: bucket_value a value vector over buckets
		@param: card_value a vector in which to store the output values over
				private hands
		'''
		pass


	def start_computation(self, pot_sizes):
		''' Initializes the value calculator with the pot size of each state that
			we are going to evaluate.
			During continual re-solving, there is one pot size for each
			initial state of the second betting round (before board cards are dealt).
			? at this point betting round ends ?
		@param pot_sizes a vector of pot sizes
		'''
		pass


	def get_value(self, ranges, values):
		''' Gives the predicted counterfactual values at each evaluated state,
			given input ranges.
		@{start_computation} must be called first. Each state to be evaluated must
				be given in the same order that pot sizes were given for that function.
				Keeps track of iterations internally, so should be called exactly
				once for every iteration of continual re-solving.
		@param: ranges An (N,2,K) tensor, where N is the number of states evaluated
				(must match input to @{start_computation}), 2 is the number of players,
				and K is the number of private hands. Contains N sets of 2 range vectors.
		@param: values an (N,2,K) tensor in which to store the N sets of 2 value vectors
				which are output
		'''
		pass
		# initializing data structures
		# handling pot feature for the nn
		# we need to find if we need remember something in this iteration
		# first iter that we need to remember something - we need to init data structures
		# computing bucket range in next street for both players at once
		# eliminating division by zero
		# using nn to compute values
		# computing value in the next round
		# normalizing values back according to the orginal range sum
		# remembering the values for the next round
		# translating bucket values back to the card values


	def get_value_on_board(self, board, values):
		''' Gives the average counterfactual values on the given board
			across previous calls to @{get_value}.
			Used to update opponent counterfactual values during re-solving
			after board cards are dealt.
		@param: board a non-empty vector of board cards
		@param: values a tensor in which to store the values
		'''
		pass
		# check if we have evaluated correct number of iterations


	def _prepare_next_round_values(self):
		''' Normalizes the counterfactual values remembered between @{get_value}
			calls so that they are an average rather than a sum.
		'''
		pass
		# do nothing if already prepared
		# eliminating division by zero




#
