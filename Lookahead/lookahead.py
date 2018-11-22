
'''
	A depth-limited lookahead of the game tree used for re-solving.
'''
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
		self.reconstruction_opponent_cfvs = None
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


	def resolve_first_node(self, player_range, opponent_range):
		''' Re-solves the lookahead using input ranges.
			Uses the input range for the opponent instead of a gadget range,
			so only appropriate for re-solving the root node of the game tree
			(where ranges are fixed).
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_range a range vector for the opponent
		'''
		self.layers[0].ranges_data[ : , : , : , : , 0, : ] = player_range.copy()
		self.layers[0].ranges_data[ : , : , : , : , 1, : ] = opponent_range.copy()
		self._compute()


	def resolve(self, player_range, opponent_cfvs):
		''' Re-solves the lookahead using an input range for the player and
			the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_cfvs a vector of cfvs achieved by the opponent
				before re-solving
		'''
		assert(player_range is not None)
		assert(opponent_cfvs is not None)
		self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)
		self.layers[0].ranges_data[ : , : , : , 0, : ] = player_range.copy()
		self.reconstruction_opponent_cfvs = opponent_cfvs
		self._compute()


	def _compute(self):
		''' Re-solves the lookahead.
		'''
		# 1.0 main loop
		for iter in range(arguments.cfr_iters):
			self._set_opponent_starting_range(iter)
			self._compute_current_strategies()
			self._compute_ranges()
			self._compute_update_average_strategies(iter)
			self._compute_terminal_equities()
			self._compute_cfvs()
			self._compute_regrets()
			self._compute_cumulate_average_cfvs(iter)
		# 2.0 at the end normalize average strategy
		self._compute_normalize_average_strategies()
		# 2.1 normalize root's CFVs
		self._compute_normalize_average_cfvs()


	def _compute_current_strategies(self):
		''' Uses regret matching to generate the players' current strategies.
		'''
		for d in range(1,self.depth):
			layer = self.layers[d]
			layer.positive_regrets_data = layer.regrets_data.copy()
			layer.positive_regrets_data = np.clip(layer.positive_regrets_data, self.regret_epsilon, constants.max_number)
			# 1.0 set regret of empty actions to 0
			layer.positive_regrets_data *= layer.empty_action_mask
			# 1.1  regret matching
			# note that the regrets as well as the CFVs have switched player indexing
			layer.regrets_sum = np.sum(layer.positive_regrets_data, axis=0, keepdims=True)
			layer.current_strategy_data = layer.positive_regrets_data / (layer.regrets_sum * np.ones_like(layer.current_strategy_data))


	def _compute_ranges(self):
		''' Using the players' current strategies, computes their
			probabilities of reaching each state of the lookahead.
		'''
		PC, HC, batch_size = constants.players_count, game_settings.hand_count, self.batch_size
		for d in range(0, self.depth-1):
			next_layer, layer, parent, grandparent = self.layers[d+1], self.layers[d], self.layers[d-1], self.layers[d-2]
			if d > 0:
				p_num_terminal_actions = parent.terminal_actions_count
				parent_num_bets = parent.bets_count
			elif d == 0:
				p_num_terminal_actions = 0
				parent_num_bets = 1
			if d > 1:
				gp_num_nonallin_bets = grandparent.nonallinbets_count
				gp_num_terminal_actions = grandparent.terminal_actions_count
			else:
				gp_num_nonallin_bets = 1
				gp_num_terminal_actions = 0
			# copy the ranges of inner nodes and transpose (np.transpose - swaps axis: 1dim <-> 2 dim)
			layer.inner_nodes = np.transpose(layer.ranges_data[ p_num_terminal_actions: , :gp_num_nonallin_bets , : , : , : , : ], [0,2,1,3,4,5])
			super_view = layer.inner_nodes
			super_view = super_view.reshape([1, parent_num_bets, -1, batch_size, PC, HC])
			super_view = super_view * np.ones_like(next_layer.ranges_data)
			next_level_strategies = next_layer.current_strategy_data
			next_layer.ranges_data = super_view.copy() # .reshape(next_level_ranges.shape)
			# multiply the ranges of the acting player by his strategy
			next_layer.ranges_data[ : , : , : , : , layer.acting_player, : ] *= next_level_strategies


	def _compute_update_average_strategies(self, iter):
		''' Updates the players' average strategies with their current strategies.
		@param: iter the current iteration number of re-solving
		'''
		if iter > arguments.cfr_skip_iters:
			# no need to go through layers since we care for the average strategy only in the first node anyway
			# note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
			self.layers[1].average_strategies_data += self.layers[1].current_strategy_data


	def _compute_terminal_equities_terminal_equity(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at each lookahead state which is a terminal state of the game.
		'''
		HC = game_settings.hand_count
		for d in range(1, self.depth):
			layer = self.layers[d]
			if d > 1 or self.first_call_terminal:
				if self.tree.street != constants.streets_count:
					self.ranges_data_call[ layer.term_call_indices[0]:layer.term_call_indices[1] ] = layer.ranges_data[1][-1].copy()
				else:
					self.ranges_data_call[ layer.term_call_indices[0]:layer.term_call_indices[1] ] = layer.ranges_data[1].reshape(self.ranges_data_call[ layer.term_call_indices[0]:layer.term_call_indices[1] ].shape)
			self.ranges_data_fold[ layer.term_fold_indices[0]:layer.term_fold_indices[1] ] = layer.ranges_data[0].reshape(self.ranges_data_fold[ layer.term_fold_indices[0]:layer.term_fold_indices[1] ].shape)
		self.terminal_equity.call_value(self.ranges_data_call.reshape([-1,HC]), self.cfvs_data_call.reshape([-1,HC]))
		self.terminal_equity.fold_value(self.ranges_data_fold.reshape([-1,HC]), self.cfvs_data_fold.reshape([-1,HC]))

		for d in range(1,self.depth):
			layer = self.layers[d]
			if self.tree.street != constants.streets_count:
				if game_settings.nl and (d > 1 or self.first_call_terminal):
					layer.cfvs_data[1][-1] = self.cfvs_data_call[ layer.term_call_indices[0]:layer.term_call_indices[1] ].copy()
			else:
				if d > 1 or self.first_call_terminal:
					layer.cfvs_data[1] = self.cfvs_data_call[ layer.term_call_indices[0]:layer.term_call_indices[1] ].reshape(layer.cfvs_data[1].shape).copy()
			layer.cfvs_data[0] = self.cfvs_data_fold[ layer.term_fold_indices[0]:layer.term_fold_indices[1] ].reshape(layer.cfvs_data[0].shape).copy()

			# correctly set the folded player by mutliplying by -1
			fold_mutliplier = -1 if layer.acting_player == constants.players.P1 else 1
			layer.cfvs_data[ 0, : , : , : , 0, : ] *= fold_mutliplier
			layer.cfvs_data[ 0, : , : , : , 1, : ] *= -fold_mutliplier


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
				if layer.ranges_data[2].shape[0] > 1 or (d == 1 and self.first_call_transition) or not game_settings.nl:
					p_start, p_end = 0, -1 # parent indices
					if d == 1:
						p_start, p_end = 0, 1 # parent indices
					elif not game_settings.nl:
						p_start, p_end = 0, layer.ranges_data.shape[1] # parent indices
					self.next_street_boxes_outputs[ layer.indices[0]:layer.indices[1] , : , : , : ] = layer.ranges_data[ 1, p_start:p_end, : , : , : , : ].copy()

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
				if layer.ranges_data[1].shape[0] > 1 or (d == 1 and self.first_call_transition) or not game_settings.nl:
					p_start, p_end = 0, -1 # parent indices
					if d == 1:
						p_start, p_end = 0, 1 # parent indices
					elif not game_settings.nl:
						p_start, p_end = 1, layer.cfvs_data.shape[1] # parent indices
					layer.cfvs_data[ 1, p_start:p_end , : , : , : , : ] = self.next_street_boxes_outputs[ layer.indices[0]:layer.indices[1], : , : , : ].copy()


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
			self.layers[d].cfvs_data *= self.layers[d].pot_size


	def _compute_cfvs(self):
		''' Using the players' reach probabilities and terminal counterfactual
			values, computes their cfvs at all states of the lookahead.
		'''
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1]
			gp_num_terminal_actions = self.layers[d-2].terminal_actions_count if d > 1 else 0
			ggp_num_nonallin_bets = self.layers[d-3].nonallinbets_count if d > 2 else 1
			#
			layer.cfvs_data[ : , : , : , : , 0, : ] *= layer.empty_action_mask
			layer.cfvs_data[ : , : , : , : , 1, : ] *= layer.empty_action_mask
			layer.placeholder_data = layer.cfvs_data.copy()
			# player indexing is swapped for cfvs
			layer.placeholder_data[ : , : , : , : , layer.acting_player, : ] *= layer.current_strategy_data
			layer.regrets_sum = np.sum(layer.placeholder_data, axis=0, keepdims=True)
			# use a swap placeholder to change [[1,2,3], [4,5,6]] into [[1,2], [3,4], [5,6]]
			parent.swap_data = layer.regrets_sum.copy().reshape(parent.swap_data.shape)
			parent.cfvs_data[ gp_num_terminal_actions: , :ggp_num_nonallin_bets , : , : , : , : ] = np.transpose(parent.swap_data, [0,2,1,3,4,5]).copy() # ? - transpose(2,3))


	def _compute_cumulate_average_cfvs(self, iter):
		''' Updates the players' average counterfactual values with their
			cfvs from the current iteration.
		@param: iter the current iteration number of re-solving
		'''
		if iter > arguments.cfr_skip_iters:
			self.layers[0].average_cfvs_data += self.layers[0].cfvs_data
			self.layers[1].average_cfvs_data += self.layers[1].cfvs_data


	def _compute_normalize_average_strategies(self):
		''' Normalizes the players' average strategies.
			Used at the end of re-solving so that we can track
			un-normalized average strategies, which are simpler to compute.
		'''
		# using regrets_sum as a placeholder container
		player_avg_strategy = self.layers[1].average_strategies_data
		player_avg_strategy_sum = self.layers[1].regrets_sum
		self.layers[1].regrets_sum = np.sum(self.layers[1].average_strategies_data, axis=0, keepdims=True)
		self.layers[1].average_strategies_data /= self.layers[1].regrets_sum * np.ones_like(player_avg_strategy)
		# if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
		# it sums to one -> now we set to always fold
		player_avg_strategy[0][ player_avg_strategy[0] != player_avg_strategy[0] ] = 1
		player_avg_strategy[ player_avg_strategy != player_avg_strategy ] = 0


	def _compute_normalize_average_cfvs(self):
		''' Normalizes the players' average counterfactual values.
			Used at the end of re-solving so that we can track
			un-normalized average cfvs, which are simpler to compute.
		'''
		self.layers[0].average_cfvs_data /= (arguments.cfr_iters - arguments.cfr_skip_iters)


	def _compute_regrets(self):
		''' Using the players' counterfactual values, updates their
			total regrets for every state in the lookahead.
		'''
		HC, batch_size = game_settings.hand_count, self.batch_size
		for d in range(self.depth-1, 0, -1):
			layer, parent = self.layers[d], self.layers[d-1] # current layer, parent layer
			gp_num_terminal_actions = self.layers[d-2].terminal_actions_count if d > 1 else 0
			gp_num_bets = self.layers[d-2].bets_count if d > 1 else 1
			ggp_num_nonallin_bets = self.layers[d-3].nonallinbets_count if d > 2 else 1
			# current_regrets = self.current_regrets_data[d]
			current_regrets = layer.cfvs_data[ : , : , : , : , layer.acting_player, : ].copy().reshape(layer.current_regrets_data.shape) # ? - no need reshape?
			parent_inner_nodes = parent.inner_nodes_p1
			parent_inner_nodes = np.transpose(parent.cfvs_data[ gp_num_terminal_actions: , :ggp_num_nonallin_bets, : , : , layer.acting_player, : ], [0,2,1,3,4])
			parent_inner_nodes = parent_inner_nodes.reshape([1, gp_num_bets, -1, batch_size, HC])
			parent_inner_nodes = parent_inner_nodes * np.ones_like(current_regrets)
			current_regrets -= parent_inner_nodes
			layer.regrets_data += current_regrets
			# (CFR+)
			layer.regrets_data = np.clip(layer.regrets_data, 0, constants.max_number)


	def get_results(self):
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
		actions_count = self.layers[1].average_strategies_data.shape[0]
		PC, HC, AC, batch_size = constants.players_count, game_settings.hand_count, actions_count, self.batch_size
		out = LookaheadResults()
		# 1.0 average strategy
		# [actions x range]
		# lookahead already computes the averate strategy we just convert the dimensions
		out.strategy = self.layers[1].average_strategies_data.reshape([-1,batch_size,HC]).copy()
		# 2.0 achieved opponent's CFVs at the starting node
		out.achieved_cfvs = self.layers[0].average_cfvs_data.reshape([batch_size,PC,HC])[0].copy()
		# 3.0 CFVs for the acting player only when resolving first node
		if self.reconstruction_opponent_cfvs is not None:
			out.root_cfvs = None
		else:
			out.root_cfvs = self.layers[0].average_cfvs_data.reshape([batch_size,PC,HC])[ : , 1 , : ].copy()
			# swap cfvs indexing
			out.root_cfvs_both_players = self.layers[0].average_cfvs_data.reshape([batch_size,PC,HC]).copy()
			out.root_cfvs_both_players[ : , 1 , : ] = self.layers[0].average_cfvs_data.reshape([batch_size,PC,HC])[ : , 0 , : ].copy()
			out.root_cfvs_both_players[ : , 0 , : ] = self.layers[0].average_cfvs_data.reshape([batch_size,PC,HC])[ : , 1 , : ].copy()
		# 4.0 children CFVs
		# [actions x range]
		out.children_cfvs = self.layers[1].average_cfvs_data[ : , : , : , : , 0, : ].copy().reshape([-1,HC])
		# IMPORTANT divide average CFVs by average strategy in here
		scaler = self.layers[1].average_strategies_data.reshape([-1,batch_size,HC]).copy()
		range_mul = self.layers[0].ranges_data[ : , : , : , : , 0, : ].reshape([1,batch_size,HC]).copy()
		range_mul = range_mul * np.ones_like(scaler)
		scaler = scaler * range_mul
		scaler = np.sum(scaler, axis=2, keepdims=True) * np.ones_like(range_mul)
		scaler = scaler * (arguments.cfr_iters - arguments.cfr_skip_iters)
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
		if self.reconstruction_opponent_cfvs is not None:
			# note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
			opponent_range = self.reconstruction_gadget.compute_opponent_range(self.layers[0].cfvs_data[ : , : , : , : , 0 , : ], iteration)
			self.layers[0].ranges_data[ : , : , : , : , 1 , : ] = opponent_range.copy()




#
