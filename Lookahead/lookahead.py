
'''
	A depth-limited lookahead of the game tree used for re-solving.
'''
import time
import numpy as np

from Lookahead.lookahead_builder import LookaheadBuilder
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.cfrd_gadget import CFRDGadget
from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from helper_classes import LookaheadResults

class Lookahead():
	def __init__(self, terminal_equity, batch_size):
		self.builder = LookaheadBuilder(self)
		self.terminal_equity = terminal_equity
		self.batch_size = batch_size


	def build_lookahead(self, tree):
		''' Constructs the lookahead from a game's public tree.
			Must be called to initialize the lookahead.
		@param: tree a public tree
		'''
		self.builder.build_from_tree(tree)


	def reset(self):
		self.builder.reset()


	def resolve(self, player_range, opponent_range=None, opponent_cfvs=None):
		if opponent_range is not None and opponent_cfvs is not None: raise('only 1 var can be passed')
		if opponent_range is None and opponent_cfvs is None: raise('one of those vars must be passed')
		# can be cfvs or range
		self.layers[0].ranges[ : , : , : , : , 0, : ] = player_range.copy()
		if opponent_cfvs is None:
			self.layers[0].ranges[ : , : , : , : , 1, : ] = opponent_range.copy()
			self._compute(reconstruct_opponent_cfvs=False)
		else:
			self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)
			self._compute(reconstruct_opponent_cfvs=True)


	def _compute(self, reconstruct_opponent_cfvs):
		''' Re-solves the lookahead.
		'''
		# 1.0 main loop
		time_arr = np.zeros([8,arguments.cfr_iters])
		t0 = time.time()
		for iter in range(arguments.cfr_iters):
			if reconstruct_opponent_cfvs:
				self._set_opponent_starting_range(iter)
			time_arr[0,iter] = time.time() - t0; t0 = time.time()
			self._compute_current_strategies()
			time_arr[1,iter] = time.time() - t0; t0 = time.time()
			self._compute_ranges()
			time_arr[2,iter] = time.time() - t0; t0 = time.time()
			if iter > arguments.cfr_skip_iters:
				self._compute_update_average_strategies(iter)
			time_arr[3,iter] = time.time() - t0; t0 = time.time()
			self._compute_terminal_equities()
			time_arr[4,iter] = time.time() - t0; t0 = time.time()
			self._compute_cfvs()
			time_arr[5,iter] = time.time() - t0; t0 = time.time()
			self._compute_regrets()
			time_arr[6,iter] = time.time() - t0; t0 = time.time()
			if iter > arguments.cfr_skip_iters:
				self._compute_cumulate_average_cfvs(iter)
			time_arr[7,iter] = time.time() - t0; t0 = time.time()
		print('times:', np.array2string(np.sum(time_arr, axis=1), suppress_small=True, precision=3))
		# 2.0 at the end normalize average strategy
		self._compute_normalize_average_strategies()
		# 2.1 normalize root's CFVs
		self._compute_normalize_average_cfvs()


	def _compute_current_strategies(self):
		''' Uses regret matching to generate the players' current strategies.
		'''
		for d in range(1,self.depth):
			layer = self.layers[d]
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			positive_regrets = layer.regrets.copy()
			positive_regrets = np.clip(positive_regrets, self.regret_epsilon, constants.max_number)
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
		PC, HC, batch_size = constants.players_count, game_settings.hand_count, self.batch_size
		for d in range(0, self.depth-1):
			next_layer, layer, parent, grandparent = self.layers[d+1], self.layers[d], self.layers[d-1], self.layers[d-2]
			p_num_terminal_actions = parent.num_terminal_actions if d > 0 else 0
			parent_num_bets = parent.num_bets if d > 0 else 1
			gp_num_nonallin_bets = grandparent.num_nonallin_bets if d > 1 else 1
			gp_num_terminal_actions = grandparent.num_terminal_actions if d > 1 else 0
			# copy the ranges of inner nodes and transpose (np.transpose - swaps axis: 1dim <-> 2 dim)
			# array slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-1}, NAB{d-2}, NTNAN{d-2}, b, P, I]
			# [B{d-1}, NTNAN{d-2}, NAB{d-2}, b, P, I] = [B{d-1}, NAB{d-2}, NTNAN{d-2}, b, P, I]
			next_layer_ranges = np.transpose(layer.ranges[ p_num_terminal_actions: , :gp_num_nonallin_bets , : , : , : , : ], [0,2,1,3,4,5])
			# [ 1, B{d-1}, NTNAN{d-2} x NAB{d-2}, b, P, I] = [B{d-1}, NTNAN{d-2}, NAB{d-2}, b, P, I]
			# [ 1, B{d-1}, NTNAN{d-2} x NAB{d-2}, b, P, I] is the same as [ 1, B{d-1}, NTNAN{d-1}, b, P, I]
			next_layer_ranges = next_layer_ranges.reshape([1, parent_num_bets, -1, batch_size, PC, HC])
			# [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I] = [ 1, B{d-1}, NTNAN{d-1}, b, P, I] * [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I]
			next_layer_ranges = next_layer_ranges * np.ones_like(next_layer.ranges)
			# [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I] = [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I]
			next_layer.ranges = next_layer_ranges.copy()
			# multiply the ranges of the acting player by his strategy
			# [ A{d}, B{d-1}, NTNAN{d-1}, b, P, I] *= [ A{d}, B{d-1}, NTNAN{d-1}, b, I]
			next_layer.ranges[ : , : , : , : , layer.acting_player, : ] *= next_layer.current_strategy


	def _compute_update_average_strategies(self, iter):
		''' Updates the players' average strategies with their current strategies.
		@param: iter the current iteration number of re-solving
		'''
		# no need to go through layers since we care for the average strategy only in the first node anyway
		# note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
		# [ A{0}, 1, 1, b, I] += [ A{0}, 1, 1, b, I]
		self.layers[1].strategies_avg += self.layers[1].current_strategy


	def _compute_terminal_equities_terminal_equity(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at each lookahead state which is a terminal state of the game.
		'''
		HC, PC, batch_size = game_settings.hand_count, constants.players_count, self.batch_size
		call_ranges = np.zeros([self.num_term_call_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		fold_ranges = np.zeros([self.num_term_fold_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		call_cfvs = np.zeros([self.num_term_call_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		fold_cfvs = np.zeros([self.num_term_fold_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		for d in range(1, self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_terminal:
				if self.tree.street != constants.streets_count:
					# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1, -1] -> [NTNAN{d-2}, b, P, I]
					call_ranges[ layer.term_call_idx[0]:layer.term_call_idx[1] ] = layer.ranges[1, -1]
				else:
					# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
					# reshape: [B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2} x NTNAN{d-2}, b, P, I] = [NTN{d-1}, b, P, I]
					call_ranges[ layer.term_call_idx[0]:layer.term_call_idx[1] ] = layer.ranges[1].reshape([-1, batch_size, PC, HC])
			# slicing: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [0] -> [B{d-2}, NTNAN{d-2}, b, P, I]
			# reshape: [B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2} x NTNAN{d-2}, b, P, I] = [NTN{d-1}, b, P, I]
			fold_ranges[ layer.term_fold_idx[0]:layer.term_fold_idx[1] ] = layer.ranges[0].reshape([-1, batch_size, PC, HC])
		# cfvs = dot(ranges, matrix)
		# [TN x b x P, I] = dot( [TN x b x P, I], [I,I] )
		self.terminal_equity.call_value( call_ranges.reshape([-1,HC]), call_cfvs.reshape([-1,HC]) )
		self.terminal_equity.fold_value( fold_ranges.reshape([-1,HC]), fold_cfvs.reshape([-1,HC]) )
		#
		for d in range(1,self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_terminal:
				if self.tree.street != constants.streets_count and game_settings.nl:
					# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1, -1] -> [NTNAN{d-2}, b, P, I]
					layer.cfvs[1][-1] = call_cfvs[ layer.term_call_idx[0]:layer.term_call_idx[1] ]
				else:
					# cfvs: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
					layer.cfvs[1] = call_cfvs[ layer.term_call_idx[0]:layer.term_call_idx[1] ].reshape(layer.cfvs[1].shape)
			# cfvs: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] [1] -> [B{d-2}, NTNAN{d-2}, b, P, I]
			layer.cfvs[0] = fold_cfvs[ layer.term_fold_idx[0]:layer.term_fold_idx[1] ].reshape(layer.cfvs[0].shape)
			# correctly set the folded player by mutliplying by -1
			fold_mutliplier = -1 if layer.acting_player == constants.players.P1 else 1
			# # [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= scalar
			layer.cfvs[ 0, : , : , : , 0, : ] *= fold_mutliplier
			layer.cfvs[ 0, : , : , : , 1, : ] *= -fold_mutliplier


	def _compute_terminal_equities_next_street_box(self):
		''' Using the players' reach probabilities, calls the neural net to
			compute the players' counterfactual values at the depth-limited
			states of the lookahead.
		'''
		PC, HC = constants.players_count, game_settings.hand_count
		assert(self.tree.street == 1)
		if self.num_pot_sizes == 0:
			return
		for d in range(1,self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_transition:
				# if there's only 1 parent, then it should've been an all in, so skip this next_street_box calculation
				if layer.ranges[2].shape[0] > 1 or (d == 1 and self.first_call_transition) or not game_settings.nl:
					# parent indices
					if d == 1:  			   p_start, p_end = 0, 1
					elif not game_settings.nl: p_start, p_end = 0, layer.ranges.shape[1]
					else: 					   p_start, p_end = 0, -1
					self.next_street_boxes_outputs[ layer.indices[0]:layer.indices[1] , : , : , : ] = layer.ranges[ 1, p_start:p_end, : , : , : , : ].copy()

		if self.tree.current_player == constants.players.P2:
			self.next_street_boxes_inputs = self.next_street_boxes_outputs.copy()
		else:
			self.next_street_boxes_inputs[ : , : , 0, : ] = self.next_street_boxes_outputs[ : , : , 1, : ].copy()
			self.next_street_boxes_inputs[ : , : , 1, : ] = self.next_street_boxes_outputs[ : , : , 0, : ].copy()

		if self.tree.street == 1:
		    self.next_street_boxes.get_value_aux(self.next_street_boxes_inputs.reshape([-1,PC,HC]), self.next_street_boxes_outputs.reshape([-1,PC,HC]), self.next_board_idx)
		else:
			self.next_street_boxes.get_value(self.next_street_boxes_inputs.reshape([-1,PC,HC]), self.next_street_boxes_outputs.reshape([-1,PC,HC]), self.next_board_idx)

		# now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
		if self.tree.current_player == constants.players.P2:
			self.next_street_boxes_inputs = self.next_street_boxes_outputs.copy()
			self.next_street_boxes_outputs[ : , : , 0, : ] = self.next_street_boxes_inputs[ : , : , 1, : ].copy()
			self.next_street_boxes_outputs[ : , : , 1, : ] = self.next_street_boxes_inputs[ : , : , 0, : ].copy()

		for d in range(1, self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_transition:
				if layer.ranges[1].shape[0] > 1 or (d == 1 and self.first_call_transition) or not game_settings.nl:
					# parent indices
					if d == 1:				   p_start, p_end = 0, 1
					elif not game_settings.nl: p_start, p_end = 1, layer.cfvs.shape[1]
					else: 					   p_start, p_end = 0, -1
					layer.cfvs[ 1, p_start:p_end , : , : , : , : ] = self.next_street_boxes_outputs[ layer.indices[0]:layer.indices[1], : , : , : ].copy()


	def get_chance_action_cfv(self, action, board):
		''' Gives the average counterfactual values for the opponent during
			re-solving after a chance event
			(the betting round changes and more cards are dealt).
			Used during continual re-solving to track opponent cfvs.
			The lookahead must first be re-solved with
			@{resolve} or @{resolve_first_node}.
		@param: action_index the action taken by the re-solving player
				at the start of the lookahead
		@param: board a tensor of board cards, updated by the chance event
		@return a vector of cfvs
		''' # ? - can be problem with chance nodes (needs another look)
		PC, HC = constants.players_count, game_settings.hand_count
		box_outputs = self.next_street_boxes_outputs.reshape([-1,PC,HC])
		next_street_box = self.next_street_boxes
		batch_index = self.action_to_index[action]
		assert(batch_index is not None)
		pot_mult = self.next_round_pot_sizes[batch_index]
		if box_outputs is None:
			assert(False)
		next_street_box.get_value_on_board(board, box_outputs)
		out = box_outputs[batch_index][self.tree.current_player]
		out *= pot_mult
		return out


	def _compute_terminal_equities(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at all terminal states of the lookahead.
			These include terminal states of the game and depth-limited states.
		'''
		if self.tree.street != constants.streets_count:
			self._compute_terminal_equities_next_street_box()
		self._compute_terminal_equities_terminal_equity()
		# multiply by pot scale factor
		for d in range(1, self.depth):
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			self.layers[d].cfvs *= self.layers[d].pot_size


	def _compute_cfvs(self):
		''' Using the players' reach probabilities and terminal counterfactual
			values, computes their cfvs at all states of the lookahead.
		'''
		PC, HC, batch_size = constants.players_count, game_settings.hand_count, self.batch_size
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1]
			num_gp_terminal_actions = self.layers[d-2].num_terminal_actions if d > 1 else 0
			num_ggp_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			layer.cfvs[ : , : , : , : , 0, : ] *= layer.empty_action_mask
			layer.cfvs[ : , : , : , : , 1, : ] *= layer.empty_action_mask
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			placeholder_data = layer.cfvs.copy()
			# player indexing is swapped for cfvs
			# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] *= [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			placeholder_data[ : , : , : , : , layer.acting_player, : ] *= layer.current_strategy
			# [ 1, B{d-2}, NTNAN{d-2}, b, P, I] = [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			regrets_sum = np.sum(placeholder_data, axis=0, keepdims=True)
			# print(layer.regrets_sum.shape, placeholder_data.shape, layer.current_strategy.shape, layer.cfvs.shape)
			# use a swap placeholder to change dimensions
			num_ggp_nonterminal_nonallin_nodes = self.layers[d-3].num_nonterminal_nonallin_nodes if d > 2 else 1
			num_gpp_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			num_gp_bets = regrets_sum.shape[1]
			# note: NTNAN{d-3} x NAB{d-3} = NTNAN{d-2}
			# reshape: [ 1, B{d-2}, NTNAN{d-2}, b, P, I] -> [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, P, I]
			swap_data = regrets_sum.reshape([num_gp_bets, num_ggp_nonterminal_nonallin_nodes, num_gpp_nonallin_bets, batch_size, PC, HC])
			parent.cfvs[ num_gp_terminal_actions: , :num_ggp_nonallin_bets , : , : , : , : ] = np.transpose(swap_data, [0,2,1,3,4,5]).copy()


	def _compute_cumulate_average_cfvs(self, iter):
		''' Updates the players' average counterfactual values with their
			cfvs from the current iteration.
		@param: iter the current iteration number of re-solving
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
		# if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
		# it sums to one -> now we set to always fold
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
		HC, batch_size = game_settings.hand_count, self.batch_size
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1] # current layer, parent layer
			gp_num_terminal_actions = self.layers[d-2].num_terminal_actions if d > 1 else 0
			gp_num_bets = self.layers[d-2].num_bets if d > 1 else 1
			ggp_num_nonallin_bets = self.layers[d-3].num_nonallin_bets if d > 2 else 1
			# reshape: [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I] -> [A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			current_regrets = layer.cfvs[ : , : , : , : , layer.acting_player, : ].copy()
			# slicing: [A{d-2}, B{d-3}, NTNAN{d-3}, b, P, I] -> [B{d-2}, NAB{d-3}, NTNAN{d-3}, b, I]
			# transpose: [B{d-2}, NAB{d-3}, NTNAN{d-3}, b, I] -> [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, I]
			parent_inner_nodes = np.transpose(parent.cfvs[ gp_num_terminal_actions: , :ggp_num_nonallin_bets, : , : , layer.acting_player, : ], [0,2,1,3,4])
			# reshape: [B{d-2}, NTNAN{d-3}, NAB{d-3}, b, I] -> [ 1, B{d-2}, NTNAN{d-3} x NAB{d-3}, b, I] = [ 1, B{d-2}, NTNAN{d-2}, b, I]
			parent_inner_nodes = parent_inner_nodes.reshape([1, gp_num_bets, -1, batch_size, HC])
			# broadcasting parent_inner_nodes: [ 1, B{d-2}, NTNAN{d-2}, b, I] -> [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			# [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I] -= [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			# [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I] += [ A{d-1}, B{d-2}, NTNAN{d-2}, b, I]
			current_regrets -= parent_inner_nodes
			layer.regrets += current_regrets
			# (CFR+)
			layer.regrets = np.clip(layer.regrets, 0, constants.max_number)


	def get_results(self, reconstruct_opponent_cfvs):
		''' Gets the results of re-solving the lookahead.
			The lookahead must first be re-solved with @{resolve} or @{resolve_first_node}.
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
		PC, HC, AC, batch_size = constants.players_count, game_settings.hand_count, num_actions, self.batch_size
		out = LookaheadResults()
		# 1.0 average strategy
		# [actions x range]
		# lookahead already computes the averate strategy we just convert the dimensions
		# reshape: [A{0}, 1, 1, b, I] -> [A{0}, b, I]
		out.strategy = self.layers[1].strategies_avg.reshape([-1,batch_size,HC]).copy()
		# 2.0 achieved opponent's CFVs at the starting node
		# reshape: [ 1, 1, 1, b, P, I] -> [b, P, I]
		out.achieved_cfvs = self.layers[0].cfvs_avg.reshape([batch_size,PC,HC])[0].copy()
		# 3.0 CFVs for the acting player only when resolving first node
		if reconstruct_opponent_cfvs:
			out.root_cfvs = None
		else:
			# reshape: [1, 1, 1, b, P, I] - > [b, P, I]
			first_layer_avg_cfvs = self.layers[0].cfvs_avg.reshape([batch_size,PC,HC])
			# slicing: [b, P, I] [1] -> [b, I]
			out.root_cfvs = first_layer_avg_cfvs[ : , 1 , : ].copy()
			# swap cfvs indexing
			# [b, P, I] <-  [1, 1, 1, b, P, I]
			out.root_cfvs_both_players = first_layer_avg_cfvs.copy()
			out.root_cfvs_both_players[ : , 1 , : ] = first_layer_avg_cfvs[ : , 0 , : ].copy()
			out.root_cfvs_both_players[ : , 0 , : ] = first_layer_avg_cfvs[ : , 1 , : ].copy()
		# 4.0 children CFVs
		# slicing and reshaping: [A{0}, 1, 1, b, P, I] -> [A{0}, b, I]
		out.children_cfvs = self.layers[1].cfvs_avg[ : , : , : , : , 0, : ].copy().reshape([-1,batch_size,HC])
		# IMPORTANT divide average CFVs by average strategy in here
		# reshape: [A{0}, 1, 1, b, I] -> [A{0}, b, I]
		scaler = self.layers[1].strategies_avg.reshape([-1,batch_size,HC]).copy()
		# slicing and reshaping: [ 1, 1, 1, b, P, I] -> [1, b, I]
		range_mul = self.layers[0].ranges[ : , : , : , : , 0, : ].reshape([1,batch_size,HC]).copy()
		# broadcasting range_mul: [1, b, I] -> [A{0}, b, I]
		scaler = scaler * range_mul
		# [A{0}, b, 1] = sum([A{0}, b, I])
		scaler = np.sum(scaler, axis=2, keepdims=True)
		# [A{0}, b, 1] *= scalar
		scaler = scaler * (arguments.cfr_iters - arguments.cfr_skip_iters)
		# broadcasting scaler: [A{0}, b, 1] -> [A{0}, b, I]
		# [A{0}, b, I] /= [A{0}, b, 1]
		out.children_cfvs = out.children_cfvs / scaler
		assert(out.strategy is not None)
		assert(out.achieved_cfvs is not None)
		assert(out.children_cfvs is not None)
		return out


	def _set_opponent_starting_range(self, iteration):
		''' Generates the opponent's range for the current re-solve iteration
			using the @{cfrd_gadget|CFRDGadget}.
		@param: iteration the current iteration number of re-solving
		'''
		# note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
		opponent_range = self.reconstruction_gadget.compute_opponent_range(self.layers[0].cfvs[ : , : , : , : , 0 , : ], iteration)
		# [1, 1, 1, P, I] = [I]
		self.layers[0].ranges[ : , : , : , : , 1 , : ] = opponent_range.copy()




#
