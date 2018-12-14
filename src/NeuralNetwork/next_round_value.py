'''
	Uses the neural net to estimate value at the end of the first betting round.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from Game.card_combinations import card_combinations

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
		self.iter = 0
		# get next round boards
		if self.nn.approximate == 'root_nodes':
			self.next_boards = card_tools.get_next_round_boards(self.current_board)
		else: # self.nn.approximate == 'leaf_nodes':
			self.next_boards = np.expand_dims(self.current_board, axis=0) # only single board
		self.next_boards_count = self.next_boards.shape[0]
		PC, BC, HC = constants.players_count, self.next_boards_count, constants.hand_count
		# init pot sizes [b, 1], where p - number of pot sizes, b - batch size (not the same as in other files)
		self.pot_sizes = pot_sizes.reshape([-1,1]) # [p,1]
		self.pot_sizes = self.pot_sizes * np.ones([self.pot_sizes.shape[0], batch_size], dtype=arguments.dtype)
		self.pot_sizes = self.pot_sizes.reshape([-1,1])
		self.batch_size, batch_size = self.pot_sizes.shape[0], self.pot_sizes.shape[0]
		# init board features [BC,69]
		self.num_board_features = constants.rank_count + constants.suit_count + constants.card_count
		self.board_features = np.zeros([BC, self.num_board_features], dtype=arguments.dtype)
		for i, board in enumerate(self.next_boards):
			self.board_features[i] = card_tools.convert_board_to_nn_feature(board)
		# init inputs and outputs to neural net
		self.next_round_inputs = np.zeros([batch_size,BC,HC*PC + 1 + self.num_board_features], dtype=arguments.dtype)
		self.next_round_values = np.zeros([batch_size,BC,PC,HC], dtype=arguments.dtype)
		# handling pot feature for nn
		# broadcasting pot_sizes: [b] -> [b,B]
		# [ b, B, P x I + 1 + 69 ] = scalar * [b] * [b,B]
		self.next_round_inputs[ : , : , PC*HC ] = (1/arguments.stack) * self.pot_sizes * np.ones([batch_size,BC], dtype=self.pot_sizes.dtype)
		# handling board feature for nn
		# reshape: [B,69] -> [1,B,69]
		nn_board_input = self.board_features.reshape([1,BC,self.num_board_features])
		# broadcasting nn_board_input: [ 1, B, 69 ] -> [ b, B, 69 ]
		# [ b, B, P x I + 1 + 69 ] = [ b, B, 69 ]
		self.next_round_inputs[ : , : , PC*HC+1: ] = nn_board_input * np.ones([batch_size,BC,self.num_board_features], dtype=arguments.dtype)
		# init board mask # [B,I]
		self.board_mask = np.zeros([BC,HC], dtype=bool)
		for i, next_board in enumerate(self.next_boards):
			self.board_mask[i] = card_tools.get_possible_hand_indexes(next_board)
		# calculate possible boards for each hand (for later to make mean of board values (sum -> normalize))
		# [I] = sum([B,I], axis=0)
		num_possible_boards = card_combinations.count_next_boards_possible_boards(self.street)
		self.sum_normalization = 1 / num_possible_boards




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
		PC, BC, HC, batch_size = constants.players_count, self.next_boards_count, constants.hand_count, self.batch_size
		assert(ranges.shape[0] == self.batch_size)
		self.iter += 1
		# init cumulative cfvs and their normalization
		if self.iter == arguments.cfr_skip_iters + 1:
			self.cumulative_norm = np.zeros([ batch_size, BC, PC ], dtype=arguments.dtype)
			self.cumulative_cfvs = np.zeros([ batch_size, BC, PC, HC ], dtype=arguments.dtype)
		# handling ranges
		# broadcasting ranges_: [ b, 1, P, I ] -> [ b, B, P, I ]
		ranges_ = ranges.reshape([batch_size,1,PC*HC])
		self.next_round_inputs[ : , : , :PC*HC ] = ranges_ * np.ones([batch_size,BC,PC*HC], dtype=arguments.dtype)
		# mask inputs
		# [b,B,PxI+1+69] *= [B,I]
		self.next_round_inputs[ : , : , :HC ] *= self.board_mask.reshape([1,BC,HC])
		self.next_round_inputs[ : , : , HC:2*HC ] *= self.board_mask.reshape([1,BC,HC])
		# sum all ranges for later to normalize ranges
		ranges_sum = np.zeros([batch_size,BC,PC], dtype=arguments.dtype)
		ranges_sum[:,:,0] = np.sum(self.next_round_inputs[ : , : , :HC ], axis=2)
		ranges_sum[:,:,1] = np.sum(self.next_round_inputs[ : , : , HC:2*HC ], axis=2)
		# save var for later on to normalize output values (swaped because of the swap at lookahead.get_results)
		values_norm = np.zeros_like(ranges_sum)
		values_norm[:,:,0] = ranges_sum[:,:,1].copy()
		values_norm[:,:,1] = ranges_sum[:,:,0].copy()
		# eliminate divide by zero
		ranges_sum[ ranges_sum == 0 ] = 1
		# normalize ranges
		self.next_round_inputs[ : , : , :HC ] /= ranges_sum[:,:,0].reshape([batch_size,BC,1])
		self.next_round_inputs[ : , : , HC:2*HC ] /= ranges_sum[:,:,1].reshape([batch_size,BC,1])
		# computing value in the next round (outputs are already masked, see neural network)
		self.nn.get_value( self.next_round_inputs.reshape([batch_size*BC,-1]), self.next_round_values.reshape([batch_size*BC,-1]) )
		# normalizing values back to original range sum
		# [b,B,P,I] *= [b,B,P,1]
		self.next_round_values *= values_norm.reshape([batch_size,BC,PC,1])
		# clip values that are more then maximum   							# 20,000     > nn_value x pot_size x 20,000 > -20,000
		max_values = 1 / self.pot_sizes.reshape([batch_size,1,1,1])  		# 1          >       nn_value x pot_size    > -1
		# [b,B,P,I] = clip([b,B,P,I], [b,1,1,1], [b,1,1,1]) 				# 1/pot_size >           nn_value           > -1/pot_size
		np.clip(self.next_round_values, -max_values, max_values, out=self.next_round_values)
		# calculate mean for each hand and return it
		# [b,P,I] = sum([b,B,P,I], axis=1) * scalar
		values[:,:,:] = np.sum(self.next_round_values, axis=1) * self.sum_normalization
		# save values in memory for later (use for get_value_on_board())
		if self.iter > arguments.cfr_skip_iters:
			self.cumulative_cfvs += self.next_round_values
			self.cumulative_norm += values_norm



	def get_stored_value_on_board(self, board):
		assert(values.shape[0] == self.batch_size)
		# remove divison by 0
		self.cumulative_norm[ self.cumulative_norm == 0 ] = 1
		# normalize cfvs
		# [b,B,P,I] /= [b,B,P,1]
		self.cumulative_cfvs /= np.expand_dims(self.cumulative_norm, axis=-1)
		# return cfvs for particular board
		if self.nn.approximate == 'root_nodes':
			for i, next_board in enumerate(self.next_boards):
				if (board == next_board).all():
					return self.cumulative_cfvs[:,i,:,:]
		else: # self.nn.approximate == 'leaf_nodes': (only one board saved)
			# TODO: how to transition to next state if you don't have next board saved?
			# load from next street (root nodes) neural net?
			mask = card_tools.get_possible_hand_indexes(board)
			self.cumulative_cfvs *= mask
			return self.cumulative_cfvs[:,0,:,:]




#
