
'''
	A depth-limited lookahead of the game tree used for re-solving.
'''
import time
import numpy as np
from numba import njit

from Lookahead.lookahead_builder import LookaheadBuilder
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.cfrd_gadget import CFRDGadget
from Settings.arguments import arguments
from Settings.constants import constants
from helper_classes import LookaheadResults

class Lookahead():
	def __init__(self, terminal_equity, batch_size):
		self.builder = LookaheadBuilder(self)
		self.terminal_equity = terminal_equity
		self.batch_size = batch_size
		# build lookahead
		self.builder.build_from_tree(tree)


	def get_results(self, reconstruct_opponent_cfvs):
		''' Gets the results of re-solving the lookahead.
		@return a table containing the fields:
				* `strategy`: an (A,K) tensor containing the re-solve player's
				strategy at the root of the lookahead, where
				A is the number of actions and K is the range size
				* `achieved_cfvs`: a vector of the opponent's
				average counterfactual values at the root of the lookahead
				* `children_cfvs`: an (A,K) tensor of opponent
				average counterfactual values after each action
				that the re-solve player can take at the root of the lookahead
		'''
		num_actions = self.layers[1].strategies_avg.shape[0]
		PC, HC, AC, batch_size = constants.players_count, constants.hand_count, num_actions, self.batch_size
		P1, P2 = constants.players.P1, constants.players.P2
		out = LookaheadResults()
		# next street CFV's
		try:
			out.next_street_cfvs = self.cfvs_approximator.get_stored_cfvs_of_all_next_round_boards()
			out.next_boards = self.cfvs_approximator.next_boards
			out.action_to_index = self.action_to_index
			out.next_round_pot_sizes = self.next_round_pot_sizes
		except:
			print('WARNING: THERE ARE NO NODES THAT NEEDS APPROXIMATION (lookahead.cfvs_approximator is not defined)')
		# save actions
		out.actions = self.tree.actions
		# 1.0 average strategy
		# [actions x range]
		# lookahead already computes the averate strategy we just convert the dimensions
		# reshape: [A{0}, 1, 1, b, I] -> [A{0}, b, I]
		out.strategy = self.layers[1].strategies_avg.reshape([-1,batch_size,HC]).copy()
		# 2.0 achieved opponent's CFVs at the starting node
		# reshape: [ 1, 1, 1, b, P, I] -> [b, P, I]
		out.achieved_cfvs = self.layers[0].cfvs_avg.reshape([batch_size,PC,HC]).copy()
		# 3.0 CFVs for the acting player only when resolving first node
		if reconstruct_opponent_cfvs:
			out.root_cfvs = None
		else:
			# reshape: [1, 1, 1, b, P, I] - > [b, P, I]
			first_layer_avg_cfvs = self.layers[0].cfvs_avg.reshape([batch_size,PC,HC])
			# slicing: [b, P, I] [1] -> [b, I]
			out.root_cfvs = first_layer_avg_cfvs[ : , P2 , : ].copy()
			# swap cfvs indexing
			# [b, P, I] <-  [1, 1, 1, b, P, I]
			out.root_cfvs_both_players = np.zeros_like(first_layer_avg_cfvs)
			out.root_cfvs_both_players[ : , P2 , : ] = first_layer_avg_cfvs[ : , P1 , : ].copy()
			out.root_cfvs_both_players[ : , P1 , : ] = first_layer_avg_cfvs[ : , P2 , : ].copy()
		# 4.0 children CFVs
		# slicing and reshaping: [A{0}, 1, 1, b, P, I] -> [A{0}, b, I]
		out.children_cfvs = self.layers[1].cfvs_avg[ : , : , : , : , P1 , : ].reshape([-1,batch_size,HC])
		# IMPORTANT divide average CFVs by average strategy in here
		# reshape: [A{0}, 1, 1, b, I] -> [A{0}, b, I]
		strategy = self.layers[1].strategies_avg.reshape([-1,batch_size,HC])
		# slicing and reshaping: [ 1, 1, 1, b, P, I] -> [1, b, I]
		range_mul = self.layers[0].ranges[ : , : , : , : , P1 , : ].reshape([1,batch_size,HC])
		# broadcasting range_mul: [1, b, I] -> [A{0}, b, I]
		# [A{0}, b, 1] = sum([A{0}, b, I])
		scaler = np.sum(strategy * range_mul, axis=2, keepdims=True)
		# [A{0}, b, 1] *= scalar
		scaler *= arguments.cfr_iters - arguments.cfr_skip_iters
		# broadcasting scaler: [A{0}, b, 1] -> [A{0}, b, I]
		# [A{0}, b, I] /= [A{0}, b, 1]
		out.children_cfvs /= scaler
		return out


	def resolve(self, player_range, opponent_range=None, opponent_cfvs=None):
		P1, P2 = constants.players.P1, constants.players.P2
		# can be cfvs or range
		self.layers[0].ranges[ 0 , 0 , 0 , : , P1 , : ] = player_range.copy()
		if opponent_cfvs is None:
			self.layers[0].ranges[ 0 , 0 , 0 , : , P2 , : ] = opponent_range.copy()
			self._compute(reconstruct_opponent_cfvs=False)
		else: # opponent_range is None:
			self.reconstruction_gadget = CFRDGadget(self.tree.board, opponent_cfvs)
			self._compute(reconstruct_opponent_cfvs=True)


	def _compute(self, reconstruct_opponent_cfvs):
		''' Re-solves the lookahead '''
		from tqdm import tqdm
		for iter in tqdm(range(arguments.cfr_iters)):
			if reconstruct_opponent_cfvs:
				self._set_opponent_starting_range()
			self._compute_current_strategies()
			self._compute_ranges()
			if iter > arguments.cfr_skip_iters:
				self._compute_update_average_strategies()
			self._compute_cfvs()
			self._compute_expected_cfvs()
			self._compute_regrets()
			if iter > arguments.cfr_skip_iters:
				self._compute_cumulate_average_cfvs()
		# at the end normalize average strategy
		self._compute_normalize_average_strategies()
		# normalize root's CFVs
		self._compute_normalize_average_cfvs()


	def _compute_current_strategies(self):
		''' Uses regret matching to generate the players' current strategies.
		'''
		for d in range(1,self.depth):
			layer = self.layers[d]
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			positive_regrets = np.clip(layer.regrets, constants.regret_epsilon, constants.max_number)
			# 1.0 set regret of empty actions to 0
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			positive_regrets *= layer.empty_action_mask
			# 1.1  regret matching
			# note that the regrets as well as the CFVs have switched player indexing
			# [ 1, B{d-2}, NTNAN{d-2}, b, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			regrets_sum = np.sum(positive_regrets, axis=0, keepdims=True)
			# broadcasting regrets_sum: [ 1, B{d-2}, NTNAN{d-2}, b, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			# [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, I] / [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			layer.current_strategy = positive_regrets / regrets_sum


	def _compute_ranges(self):
		''' Using the players' current strategies, computes their
			probabilities of reaching each state of the lookahead.
		'''
		PC, HC, batch_size = constants.players_count, constants.hand_count, self.batch_size
		for d in range(0, self.depth-1):
			next_layer, layer, parent, grandparent = self.layers[d+1], self.layers[d], self.layers[d-1], self.layers[d-2]
			p_num_terminal_actions = parent.num_terminal_actions if d > 0 else 0
			p_num_bets = parent.num_bets if d > 0 else 1
			gp_num_nonallin_bets = grandparent.num_nonallin_bets if d > 1 else 1
			gp_num_terminal_actions = grandparent.num_terminal_actions if d > 1 else 0
			# copy the ranges of inner nodes and transpose (np.transpose - swaps axis: 1dim <-> 2 dim)
			# array slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-1}, NAB{d-2}, NTNAN{d-2}, b, P, I]
			# [B{d-1}, NTNAN{d-2}, NAB{d-2}, b, P, I] = [B{d-1}, NAB{d-2}, NTNAN{d-2}, b, P, I]
			next_layer_ranges = np.transpose(layer.ranges[ p_num_terminal_actions: , :gp_num_nonallin_bets , : , : , : , : ], [0,2,1,3,4,5])
			# [ 1, B{d-1}, NTNAN{d-2} x NAB{d-2}, b, P, I] = [B{d-1}, NTNAN{d-2}, NAB{d-2}, b, P, I]
			# [ 1, B{d-1}, NTNAN{d-2} x NAB{d-2}, b, P, I] is the same as [ 1, B{d-1}, NTNAN{d-1}, b, P, I]
			next_layer_ranges = next_layer_ranges.reshape([1, p_num_bets, -1, batch_size, PC, HC])
			# repeat next_layer_ranges: [ 1, B{d-1}, NTNAN{d-1}, b, P, I] -> [A{d}, B{d-1}, NTNAN{d-1}, b, P, I]
			# [A{d}, B{d-1}, NTNAN{d-1}, b, P, I] = [A{d}, B{d-1}, NTNAN{d-1}, b, P, I]
			next_layer.ranges = np.repeat(next_layer_ranges, next_layer.ranges.shape[0], axis=0)
			# multiply the ranges of the acting player by his strategy
			# [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I] *= [ A{d}, B{d-1}, NTNAN{d-1}, b, I]
			next_layer.ranges[ : , : , : , : , layer.acting_player, : ] *= next_layer.current_strategy


	def _compute_update_average_strategies(self):
		''' Updates the players' average strategies with their current strategies '''
		# no need to go through layers since we care for the average strategy only in the first node anyway
		# note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
		# [ A{0}, 1, 1, b, I] += [ A{0}, 1, 1, b, I]
		self.layers[1].strategies_avg += self.layers[1].current_strategy


	def _compute_expected_cfvs(self):
		''' Using the players' reach probabilities and terminal counterfactual
			values, computes their cfvs at all states of the lookahead.
		'''
		PC, HC, batch_size = constants.players_count, constants.hand_count, self.batch_size
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1]
			num_gp_terminal_actions = self.layers[d-2].num_terminal_actions if d > 1 else 0
			num_ggp_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			num_ggp_nonterminal_nonallin_nodes = self.layers[d-3].num_nonterminal_nonallin_nodes if d > 2 else 1
			num_ggp_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			# expand_dims on mask: [A{d-1}, B{d-2}, NTNAN{d-2}, b, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I]
			# broadcasting mask: [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I]
			layer.cfvs *= np.expand_dims(layer.empty_action_mask, axis=4)
			# save acting_player cfvs and use layer.cfvs as placeholder
			# player indexing is swapped for cfvs
			# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			acting_player_cfvs = layer.cfvs[ : , : , : , : , layer.acting_player, : ].copy()
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			layer.cfvs[ : , : , : , : , layer.acting_player, : ] *= layer.current_strategy
			# [ 1, B{d-2}, NTNAN{d-2}, b, P, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			expected_cfvs = np.sum(layer.cfvs, axis=0, keepdims=True)
			# leave cfvs the same as it was before
			layer.cfvs[ : , : , : , : , layer.acting_player, : ] = acting_player_cfvs
			# change dimensions
			num_gp_bets = expected_cfvs.shape[1]
			# note: NTNAN{d-3} x NAB{d-3} = NTNAN{d-2}
			# reshape: [ 1, B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, P, I]
			expected_cfvs = expected_cfvs.reshape([num_gp_bets, num_ggp_nonterminal_nonallin_nodes, num_ggp_nonallin_bets, batch_size, PC, HC])
			parent.cfvs[ num_gp_terminal_actions: , :num_ggp_nonallin_bets , : , : , : , : ] = np.transpose(expected_cfvs, [0,2,1,3,4,5])


	def _compute_cumulate_average_cfvs(self):
		''' Updates the players' average counterfactual values with their
			cfvs from the current iteration.
		'''
		# [ 1, 1, 1, b, I] += [ 1, 1, 1, b, I]
		self.layers[0].cfvs_avg += self.layers[0].cfvs
		# [ A{0}, 1, 1, b, I] += [ A{0}, 1, 1, b, I]
		self.layers[1].cfvs_avg += self.layers[1].cfvs


	def _compute_normalize_average_strategies(self):
		''' Normalizes the players' average strategies.
			Used at the end of re-solving so that we can track
			un-normalized average strategies, which are simpler to compute.
		'''
		# [ 1, 1, 1, b, I] = [A{0}, 1, 1, b, I]
		avg_strat_sum = np.sum(self.layers[1].strategies_avg, axis=0, keepdims=True)
		# broadcasting: [ 1, 1, 1, b, I] -> [A{0}, 1, 1, b, I]
		# [A{0}, 1, 1, b, I] /= [ 1, 1, 1, b, I]
		self.layers[1].strategies_avg /= avg_strat_sum
		# if the strategy is nans (zero reach), strategy does not matter but we need to make sure
		# it sums to one -> now we set to always fold
		# note: np.nan != np.nan = True, np.nan == np.nan = False
		self.layers[1].strategies_avg[0][ self.layers[1].strategies_avg[0] != self.layers[1].strategies_avg[0] ] = 1
		self.layers[1].strategies_avg[ self.layers[1].strategies_avg != self.layers[1].strategies_avg ] = 0


	def _compute_normalize_average_cfvs(self):
		''' Normalizes the players' average counterfactual values.
			Used at the end of re-solving so that we can track
			un-normalized average cfvs, which are simpler to compute.
		'''
		# [ 1, 1, 1, b, P, I] /= scalar
		self.layers[0].cfvs_avg /= (arguments.cfr_iters - arguments.cfr_skip_iters)


	def _compute_regrets(self):
		''' Using the players' counterfactual values, updates their
			total regrets for every state in the lookahead.
		'''
		HC, batch_size = constants.hand_count, self.batch_size
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1] # current layer, parent layer
			gp_num_terminal_actions = self.layers[d-2].num_terminal_actions if d > 1 else 0
			gp_num_bets = self.layers[d-2].num_bets if d > 1 else 1
			ggp_num_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			current_cfvs = layer.cfvs[ : , : , : , : , layer.acting_player, : ]
			# slicing: [A{d-2}, B{d-3}, NTNAN{d-3}, b, P, I] -> [B{d-2}, NAB{d-3}, NTNAN{d-3}, b, I]
			# transpose: [B{d-2}, NAB{d-3}, NTNAN{d-3}, b, I] -> [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, I]
			expected_cfvs = np.transpose(parent.cfvs[ gp_num_terminal_actions: , :ggp_num_nonallin_bets, : , : , layer.acting_player, : ], [0,2,1,3,4])
			# reshape: [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, I] -> [ 1, B{d-2}, NTNAN{d-3} x NAB{d-3}, b, I] = [ 1, B{d-2}, NTNAN{d-2}, b, I]
			expected_cfvs = expected_cfvs.reshape([1, gp_num_bets, -1, batch_size, HC])
			# broadcasting parent_cfvs: [ 1, B{d-2}, NTNAN{d-2}, b, I] -> [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			# [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I] += [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I] - [ 1, B{d-2}, NTNAN{d-2}, b, I]
			layer.regrets += current_cfvs - expected_cfvs
			# (CFR+)
			np.clip(layer.regrets, 0, constants.max_number, out=layer.regrets)


	def _set_opponent_starting_range(self):
		''' Generates the opponent's range for the current re-solve iteration
			using the CFRDGadget.
		'''
		P1, P2, HC = constants.players.P1, constants.players.P2, constants.hand_count
		# note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
		opponent_cfvs = self.layers[0].cfvs[ : , : , : , : , P1 , : ].reshape(HC)
		opponent_range = self.reconstruction_gadget.compute_opponent_range(opponent_cfvs)
		# [1, 1, 1, P, I] = [I]
		self.layers[0].ranges[ : , : , : , : , P2 , : ] = opponent_range


	def _compute_cfvs(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at all terminal states of the lookahead.
			These include terminal states of the game and depth-limited states.
		'''
		P1, P2, HC = constants.players.P1, constants.players.P2, constants.hand_count
		# if this is not last street and there are nodes to approximate, then approximate equity from neural network
		if self.tree.street != constants.streets_count and self.num_pot_sizes != 0:
			# store ranges of all nodes, that are transitioning to next street
			# ranges.shape = [ self.num_pot_sizes x self.batch_size, P, I ]
			ranges = self._get_ranges_from_transitioning_nodes()
			# order ranges to same order as trained examples of neural network
			if self.tree.current_player == P1:
				temp = ranges.copy()
				ranges[ : , P1, : ] = temp[ : , P2, : ]
				ranges[ : , P2, : ] = temp[ : , P1, : ]
			# use neural net to approximate cfvs
			# cfvs.shape = [ self.num_pot_sizes x self.batch_size, P, I ]
			approximated_cfvs = self.cfvs_approximator.evaluate_ranges(ranges)
			# now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
			if self.tree.current_player == P2:
				temp = approximated_cfvs.copy()
				approximated_cfvs[ : , P1, : ] = temp[ : , P2, : ]
				approximated_cfvs[ : , P2, : ] = temp[ : , P1, : ]
			# store outputs into respective nodes
			self._store_cfvs_to_transitioning_nodes(approximated_cfvs)
		# equities of all other nodes are easily computable
		# by using terminal equity/reward matrix from rules of the game
		# equities to all nodes that are terminal (game is over) are computed
		# using fold matrix (if last move was fold) and equity matrix (when all cards are shown)
		equity_matrix = self.terminal_equity.equity_matrix
		fold_matrix = self.terminal_equity.fold_matrix
		# load ranges from nodes that are terminal
		call_ranges = self._get_ranges_from_call_nodes() # [TN x b, P, I]
		fold_ranges = self._get_ranges_from_fold_nodes() # [TN x b, P, I]
		# calculate cfvs for all terminal nodes and both players
		# [TN x b x P, I] = dot_product( [TN x b x P, I], [I,I] )
		call_cfvs = np.dot(call_ranges.reshape([-1,HC]), equity_matrix)
		fold_cfvs = np.dot(fold_ranges.reshape([-1,HC]), fold_matrix)
		# no need to reshape cfvs. tensors are reshaped inside store functions
		self._store_cfvs_to_call_nodes(call_cfvs)
		self._store_cfvs_to_fold_nodes(fold_cfvs)
		# multiply all equities (from neural network and terminal equity) by pot scale factor
		for d in range(1, self.depth):
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			self.layers[d].cfvs *= self.layers[d].pot_size



	def _get_ranges_from_call_nodes(self):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		ranges = np.zeros([self.num_term_call_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		for d in range(1, self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_terminal:
				if self.tree.street != constants.streets_count:
					# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1, -1] -> [NTNAN{d-2}, b, P, I]
					ranges[ layer.term_call_idx[0]:layer.term_call_idx[1] ] = layer.ranges[1, -1]
				else:
					# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
					# reshape: [B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2} x NTNAN{d-2}, b, P, I] = [NTN{d-1}, b, P, I]
					ranges[ layer.term_call_idx[0]:layer.term_call_idx[1] ] = layer.ranges[1].reshape([-1, batch_size, PC, HC])
		return ranges.reshape([-1,PC,HC])

	def _store_cfvs_to_call_nodes(self, cfvs):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		cfvs = cfvs.reshape([self.num_term_call_nodes, batch_size, PC, HC])
		for d in range(1,self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_terminal:
				if self.tree.street != constants.streets_count:
					# allin -> call ([1] = call, [-1] = allin)
					# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1, -1] -> [NTNAN{d-2}, b, P, I]
					layer.cfvs[1][-1] = cfvs[ layer.term_call_idx[0]:layer.term_call_idx[1] ]
				else: # call ([1] = call)
					# cfvs: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
					layer.cfvs[1] = cfvs[ layer.term_call_idx[0]:layer.term_call_idx[1] ].reshape(layer.cfvs[1].shape)



	def _get_ranges_from_fold_nodes(self):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		ranges = np.zeros([self.num_term_fold_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		for d in range(1, self.depth):
			layer = self.layers[d]
			# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [0] -> [B{d-2}, NTNAN{d-2}, b, P, I]
			# reshape: [B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2} x NTNAN{d-2}, b, P, I] = [NTN{d-1}, b, P, I]
			ranges[ layer.term_fold_idx[0]:layer.term_fold_idx[1] ] = layer.ranges[0].reshape([-1, batch_size, PC, HC])
		return ranges.reshape([-1,PC,HC])

	def _store_cfvs_to_fold_nodes(self, cfvs):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		cfvs = cfvs.reshape([self.num_term_fold_nodes, batch_size, PC, HC])
		for d in range(1,self.depth):
			layer = self.layers[d]
			# cfvs: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
			layer.cfvs[0] = cfvs[ layer.term_fold_idx[0]:layer.term_fold_idx[1] ].reshape(layer.cfvs[0].shape)
			# correctly set the folded player by mutliplying by -1
			fold_mutliplier = -1 if layer.acting_player == constants.players.P1 else 1
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= scalar
			layer.cfvs[ 0, : , : , : , 0, : ] *= fold_mutliplier
			layer.cfvs[ 0, : , : , : , 1, : ] *= -fold_mutliplier



	def _get_ranges_from_transitioning_nodes(self):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		ranges = np.zeros([self.num_pot_sizes, batch_size, PC, HC], dtype=arguments.dtype)
		for d in range(1,self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_transition:
				# if there's only 1 parent, then it should've been an all in, so skip this next_street_box calculation
				num_grandparent_bets = layer.ranges[1].shape[0]
				if num_grandparent_bets > 1 or (d == 1 and self.first_call_transition):
					p_start, p_end = (0,1) if d == 1 else (0,-1) # parent indices
					# reshape: [B{d-2} - 1, NTNAN{d-2}, b, P, I] -> [(B{d-2} - 1) x NTNAN{d-2}, b, P, I]
					ranges_batch = layer.ranges[ 1, p_start:p_end, : , : , : , : ].reshape([-1, batch_size, PC, HC])
					# [sliced(PS), b, P, I] = [(B{d-2} - 1) x NTNAN{d-2}, b, P, I]
					ranges[ layer.indices[0]:layer.indices[1] , : , : , : ] = ranges_batch.copy()
		return ranges.reshape([-1,PC,HC])

	def _store_cfvs_to_transitioning_nodes(self, approximated_cfvs):
		HC, PC, batch_size = constants.hand_count, constants.players_count, self.batch_size
		approximated_cfvs = approximated_cfvs.reshape([self.num_pot_sizes, batch_size, PC, HC])
		for d in range(1, self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_transition:
				num_grandparent_bets = layer.ranges[1].shape[0]
				if num_grandparent_bets > 1 or (d == 1 and self.first_call_transition):
					p_start, p_end = (0,1) if d == 1 else (0,-1) # parent indices
					cfvs_shape = layer.cfvs[ 1, p_start:p_end , : , : , : , : ].shape
					# reshape: [sliced(p), b, P, I] -> [sliced(B{d-2}), NTNAN{d-2}, b, P, I]
					cfvs_batch = approximated_cfvs[ layer.indices[0]:layer.indices[1], : , : , : ].reshape(cfvs_shape)
					# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] = [sliced(B{d-2}), NTNAN{d-2}, b, P, I]
					layer.cfvs[ 1, p_start:p_end , : , : , : , : ] = cfvs_batch.copy()




#
