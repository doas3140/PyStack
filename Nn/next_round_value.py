'''
	Uses the neural net to estimate value at the end of the first betting round.
'''
import numpy as np

from Game.card_tools import card_tools
from Settings.arguments import arguments
from Settings.game_settings import game_settings
from Settings.constants import constants

class NextRoundValue():
	def __init__(self, value_nn, board):
		''' Creates a tensor that can translate hand ranges to bucket ranges
			on any board.
		@param: Nn.ValueNn object
		'''
		self._values_are_prepared = False
		self.nn = value_nn
		self.current_board = board
		self.street = card_tools.board_to_street(board)


	def start_computation(self, pot_sizes, batch_size):
		''' Initializes the value calculator with the pot size of each state that
			we are going to evaluate.
			During continual re-solving, there is one pot size for each
			initial state of the second betting round (before board cards are dealt).
			? at this point betting round ends ?
		@param pot_sizes a vector of pot sizes
		'''
		# get next round boards
		self.next_boards = card_tools.get_next_round_boards(self.current_board)
		self.next_boards_count = self.next_boards.shape[0]
		PC, BC, HC = constants.players_count, self.next_boards_count, game_settings.hand_count
		# init pot sizes [b, 1], where p - number of pot sizes, b - batch size (not the same as in other files)
		self.pot_sizes = pot_sizes.reshape([-1,1]) # [p,1]
		self.pot_sizes = self.pot_sizes * np.ones([self.pot_sizes.shape[0], batch_size], dtype=arguments.dtype)
		self.pot_sizes = self.pot_sizes.reshape([-1,1])
		self.batch_size, batch_size = self.pot_sizes.shape[0], self.pot_sizes.shape[0]
		# init board features [BC,69]
		self.num_board_features = game_settings.rank_count + game_settings.suit_count + game_settings.card_count
		self.board_features = np.zeros([BC, self.num_board_features], dtype=arguments.dtype)
		for i, board in enumerate(self.next_boards):
			self.board_features[i] = card_tools.convert_board_to_nn_feature(board)
		# init inputs and outputs to neural net
		self.next_round_inputs = np.zeros([batch_size,BC,HC*PC + 1 + self.num_board_features], dtype=arguments.dtype)
		self.next_round_values = np.zeros([batch_size,BC,PC,HC], dtype=arguments.dtype)
		# handling pot feature for nn
		nn_bet_input = self.pot_sizes * (1/arguments.stack)
		nn_bet_input = nn_bet_input.reshape([-1,1]) * np.ones([batch_size,BC], dtype=nn_bet_input.dtype)
		# [ b, B, P x I + 1 + 69 ] = [ b, B ]
		self.next_round_inputs[ : , : , PC*HC ] = nn_bet_input
		# handling board feature for nn
		# reshape: [B,69] -> [1,B,69]
		nn_board_input = self.board_features.reshape([1,BC,self.num_board_features])
		# broadcasting nn_board_input: [ 1, B, 69 ] -> [ b, B, 69 ]
		# [ b, B, P x I + 1 + 69 ] = [ b, B, 69 ]
		self.next_round_inputs[ : , : , PC*HC+1: ] = nn_board_input * np.ones([batch_size,BC,self.num_board_features], dtype=arguments.dtype)
		# init board mask
		# [B,I]
		self.board_mask = np.zeros([BC,HC], dtype=bool)
		for i, next_board in enumerate(self.next_boards):
			self.board_mask[i] = card_tools.get_possible_hand_indexes(next_board)
		# calculate possible boards for each hand (for later to make mean of board values (sum -> normalize))
		# [I] = sum([B,I], axis=0)
		self.mean_normalization = np.sum(self.board_mask, axis=0) / BC




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
		PC, BC, HC, batch_size = constants.players_count, self.next_boards_count, game_settings.hand_count, self.batch_size
		assert(ranges.shape[0] == self.batch_size)
		# handling ranges
		# broadcasting ranges_: [ b, 1, P, I ] -> [ b, B, P, I ]
		ranges_ = ranges.reshape([batch_size,1,PC*HC])
		self.next_round_inputs[ : , : , :PC*HC ] = ranges_ * np.ones([batch_size,BC,PC*HC], dtype=arguments.dtype)
		# mask inputs ?
		self.next_round_inputs[ : , : , :HC ] *= self.board_mask
		self.next_round_inputs[ : , : , HC:2*HC ] *= self.board_mask
		# computing value in the next round
		self.nn.get_value( self.next_round_inputs.reshape([batch_size*BC,-1]), self.next_round_values.reshape([batch_size*BC,-1]) )
		# for i in range(8):
		# 	r = self.next_round_inputs[i,:,:1326*2].reshape([-1])
		# 	# r = r[1326:]
		# 	m1 = np.zeros_like(r)
		# 	m1[ r > 0] = 1
		# 	print(m1.sum(), end=' ')
		# 	v = self.next_round_values[i,:].reshape([-1])
		# 	m = np.ones_like(v)
		# 	m[ v == 0] = 0
		# 	print(m.sum(), end=' ')
		# 	print(np.array_equal(m,m1))
		#
		# print()
		# asd()

		# apply mask (with possible boards with following cards in hand)
		# broadcasting mask: [1,B,1,I] -> [b,B,P,I]
		# [b,B,P,I] *= [b,B,P,I]
		# self.next_round_values *= self.board_mask.reshape([1,BC,1,HC])
		# calculate mean for each hand
		# [b,P,I] = sum([b,B,P,I], axis=1) / [1,1,I]
		board_values_mean = np.mean(self.next_round_values, axis=1) * self.mean_normalization.reshape([1,1,HC])
		# store it in values var # [b,P,I] = [b,P,I]
		values[:,:,:] = board_values_mean



	def get_value_on_board(self, board, values):
		''' Gives the average counterfactual values on the given board
			across previous calls to @{get_value}.
			Used to update opponent counterfactual values during re-solving
			after board cards are dealt.
		@param: board a non-empty vector of board cards
		@param: values a tensor in which to store the values
		'''
		# check if we have evaluated correct number of iterations
		assert(self.iter == arguments.cfr_iters)
		batch_size = values.shape[0]
		assert(batch_size == self.batch_size)
		self._prepare_next_round_values()
		self._bucket_value_to_card_value_on_board(board, self.counterfactual_value_memory, values)


	def _prepare_next_round_values(self):
		''' Normalizes the counterfactual values remembered between @{get_value}
			calls so that they are an average rather than a sum.
		'''
		bC = self.bucket_count
		assert(self.iter == arguments.cfr_iters)
		# do nothing if already prepared
		if self._values_are_prepared:
			return
		# eliminating division by zero
		self.range_normalization_memory[ self.range_normalization_memory == 0 ] = 1
		serialized_memory_view = self.counterfactual_value_memory.reshape([-1,bC])
		serialized_memory_view[:,:] /= self.range_normalization_memory * np.ones_like(serialized_memory_view)
		self._values_are_prepared = True




#
