'''
	Uses the neural net to estimate value at the end of the first betting round.
'''

from Nn.bucketer import Bucketer
from Game.card_tools import card_tools
from Settings.arguments import arguments
from Settings.game_settings import game_settings
from Settings.constants import constants

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
		CC, BC, bC = game_settings.card_count, self.board_count, self.bucket_count
		self.bucketer = Bucketer()
		self.bucket_count = self.bucketer.get_bucket_count()
		boards = card_tools.get_second_round_boards()
		self.board_count = boards.shape[0]
		self._range_matrix = np.zeros([CC,BC*bC], dtype=arguments.dtype)
		self._range_matrix_board_view = self._range_matrix.reshape([CC,BC,bC])
		for idx in range(1,self.board_count+1):
			board = boards[idx]
			buckets = self.bucketer.compute_buckets(board)
			class_ids = np.arange(1, self.bucket_count+1)
			class_ids = class_ids.reshape([1,bC]) * np.ones([CC,bC], dtype=class_ids.dtype)
			card_buckets = buckets.reshape([CC,1]) * np.ones([CC,bC], dtype=class_ids.dtype)
			# finding all strength classes
			# matrix for transformation from card ranges to strength class ranges
			self._range_matrix_board_view[ : , idx, : ][ class_ids == card_buckets ] = 1
		# matrix for transformation from class values to card values
		self._reverse_value_matrix = self._range_matrix.T.copy()
		# we need to div the matrix by the sum of possible boards
		# (from point of view of each hand)
		weight_constant = 1/(self.board_count - 2) # count
		self._reverse_value_matrix *= weight_constant


	def _card_range_to_bucket_range(self, card_range, bucket_range):
		''' Converts a range vector over private hands to a range vector over buckets.
		@param: card_range a probability vector over private hands
		@param: bucket_range a vector in which to store the output probabilities
				over buckets
		'''
		bucket_range[:,:] = np.dot(card_range, self._range_matrix)


	def _bucket_value_to_card_value(self, bucket_value, card_value):
		''' Converts a value vector over buckets to a value vector over private hands.
		@param: bucket_value a value vector over buckets
		@param: card_value a vector in which to store the output values over
				private hands
		'''
		card_value[:,:] = np.dot(bucket_value, self._reverse_value_matrix)


	def _bucket_value_to_card_value_on_board(self, board, bucket_value, card_value):
		''' Converts a value vector over buckets to a value vector over
			private hands given a particular set of board cards.
		@param: board a non-empty vector of board cards
		@param: bucket_value a value vector over buckets
		@param: card_value a vector in which to store the output values over
				private hands
		'''
		board_idx = card_tools.get_board_index(board)
		board_matrix = self._range_matrix_board_view[ : , board_idx, : ].T
		serialized_card_value = card_value.reshape([-1,CC])
		serialized_bucket_value = bucket_value[ : , : , board_idx, : ].copy().reshape([-1,bC])
		serialized_card_value[:,:] = np.dot(serialized_bucket_value, board_matrix)


	def start_computation(self, pot_sizes):
		''' Initializes the value calculator with the pot size of each state that
			we are going to evaluate.
			During continual re-solving, there is one pot size for each
			initial state of the second betting round (before board cards are dealt).
			? at this point betting round ends ?
		@param pot_sizes a vector of pot sizes
		'''
		self.iter = 0
		self.pot_sizes = pot_sizes.reshape([-1,1]).copy()
		self.batch_size = pot_sizes.shape[0]


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
		PC, BC = constants.players_count, self.board_count
		BS, bC = self.batch_size, self.bucket_count
		assert(ranges and values)
		assert(ranges.shape[0] == self.batch_size)
		self.iter = self.iter + 1
		if self.iter == 1:
			# initializing data structures
			self.next_round_inputs = np.zeros([BS,BC,bC*PC + 1], dtype=arguments.dtype)
			self.next_round_values = np.zeros([BS,BC,PC,bC], dtype=arguments.dtype)
			self.transposed_next_round_values = np.zeros([BS,PC,BC,bC], dtype=arguments.dtype)
			self.next_round_extended_range = np.zeros([BS,PC,BC*bC], dtype=arguments.dtype)
			self.next_round_serialized_range = self.next_round_extended_range.reshape([-1,bC])
			self.range_normalization = np.zeros([])
			self.value_normalization = np.zeros([BS,PC,BC], dtype=arguments.dtype)
			# handling pot feature for the nn
			nn_bet_input = self.pot_sizes.copy() * (1/arguments.stack)
			nn_bet_input = nn_bet_input.reshape([-1,1]) * np.ones([BS,BC], dtype=nn_bet_input.dtype)
			self.next_round_inputs[ : , : , -1 ] = nn_bet_input.copy()
		# we need to find if we need remember something in this iteration
		use_memory = self.iter > arguments.cfr_skip_iters
		if use_memory and self.iter == arguments.cfr_skip_iters + 1:
			# first iter that we need to remember something - we need to init data structures
			self.range_normalization_memory = np.zeros([BS*BC*PC,1], dtype=arguments.dtype)
			self.counterfactual_value_memory = np.zeros([BS,PC,BC,bC], dtype=arguments.dtype)
		# computing bucket range in next street for both players at once
		self._card_range_to_bucket_range( ranges.reshape([BS*PC,-1]), self.next_round_extended_range.reshape([BS*PC,-1]) )
		self.range_normalization = np.sum(self.next_round_serialized_range, axis=1, keepdims=True)
		rn_view = self.range_normalization.reshape([BS,PC,BC])
		for player in range(1,constants.players_count+1):
			self.value_normalization[ : , player, : ] = rn_view[ : , 3 - player, : ].copy()
		if use_memory:
			self.range_normalization_memory += self.value_normalization
		# eliminating division by zero
		self.range_normalization[ self.range_normalization == 0 ] = 1
		self.next_round_serialized_range /= self.range_normalization * np.ones_like(self.next_round_serialized_range)
		serialized_range_by_player = self.next_round_serialized_range.reshape([BS,PC,BC,bC])
		for player in range(1, constants.players_count+1):
			self.next_round_inputs[ : , : , (player-1)*bC+1:player*bC ] = self.next_round_extended_range[ : , player, : ].copy()
		# using nn to compute values
		serialized_inputs_view = self.next_round_inputs.reshape([BS*BC,-1])
		serialized_values_view = self.next_round_values.reshape([BS*BC,-1])
		# computing value in the next round
		self.nn.get_value(serialized_inputs_view, serialized_values_view)
		# normalizing values back according to the orginal range sum
		normalization_view = np.transpose(self.value_normalization.reshape([BS,PC,BC,1]), [0,2,1,3]) # :transpose(2,3)
		self.next_round_values *= normalization_view * np.ones_like(self.next_round_values)
		self.transposed_next_round_values = np.transpose(self.next_round_values, [0,2,1,3]).copy() # :transpose(3,2)
		# remembering the values for the next round
		if use_memory:
			self.counterfactual_value_memory += self.transposed_next_round_values
		# translating bucket values back to the card values
		self._bucket_value_to_card_value( self.transposed_next_round_values.reshape([BS*PC,-1]), values.reshape([BS*PC,-1]) )


	def get_value_on_board(self, board, values):
		''' Gives the average counterfactual values on the given board
			across previous calls to @{get_value}.
			Used to update opponent counterfactual values during re-solving
			after board cards are dealt.
		@param: board a non-empty vector of board cards
		@param: values a tensor in which to store the values
		'''
		# check if we have evaluated correct number of iterations
		assert(self.iter == arguments.cfr_iters )
		batch_size = values.shape[0]
		assert(batch_size == self.batch_size)
		self._prepare_next_round_values()
		self._bucket_value_to_card_value_on_board(board, self.counterfactual_value_memory, values)


	def _prepare_next_round_values(self):
		''' Normalizes the counterfactual values remembered between @{get_value}
			calls so that they are an average rather than a sum.
		'''
		assert(self.iter == arguments.cfr_iters )
		# do nothing if already prepared
		if self._values_are_prepared:
			return
		# eliminating division by zero
		self.range_normalization_memory[ self.range_normalization_memory == 0 ] = 1
		serialized_memory_view = self.counterfactual_value_memory.reshape([-1,bC])
		serialized_memory_view /= self.range_normalization_memory * np.ones_like(serialized_memory_view)
		self._values_are_prepared = True




#
