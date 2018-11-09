
'''
	A depth-limited lookahead of the game tree used for re-solving.
'''

from Lookahead.lookahead_builder import LookaheadBuilder
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.cfrd_gadget import CFRDGadget
from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings

class Lookahead():
	def __init__(self):
		self.builder = LookaheadBuilder(self)


	def build_lookahead(self, tree):
		''' Constructs the lookahead from a game's public tree.
			Must be called to initialize the lookahead.
		@param: tree a public tree
		'''
		self.builder.build_from_tree(tree)
		self.terminal_equity = TerminalEquity()
		self.terminal_equity.set_board(tree.board)


	def resolve_first_node(self, player_range, opponent_range):
		''' Re-solves the lookahead using input ranges.
			Uses the input range for the opponent instead of a gadget range,
			so only appropriate for re-solving the root node of the game tree
			(where ranges are fixed).
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_range a range vector for the opponent
		'''
		self.ranges_data[1][ : , : , : , 0, : ] = player_range.copy()
		self.ranges_data[1][ : , : , : , 1, : ] = opponent_range.copy()
		self._compute()


	def resolve(self, player_range, opponent_cfvs):
		''' Re-solves the lookahead using an input range for the player and
			the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_cfvs a vector of cfvs achieved by the opponent
				before re-solving
		'''
		assert(player_range)
		assert(opponent_cfvs)
		self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)
		self.ranges_data[1][ : , : , : , 0, : ] = player_range.copy()
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
		for d in range(2,self.depth+1):
		    self.positive_regrets_data[d] = self.regrets_data[d].copy()
		    self.positive_regrets_data[d] = np.clip(self.positive_regrets_data[d], self.regret_epsilon, constants.max_number)
		    # 1.0 set regret of empty actions to 0
		    self.positive_regrets_data[d] *= self.empty_action_mask[d]
		    # 1.1  regret matching
		    # note that the regrets as well as the CFVs have switched player indexing
		    self.regrets_sum[d] = np.sum(self.positive_regrets_data[d], axis=0, keepdims=True)
		    player_current_strategy = self.current_strategy_data[d]
		    player_regrets = self.positive_regrets_data[d]
		    player_regrets_sum = self.regrets_sum[d]
		    player_current_strategy = player_regrets - player_regrets_sum * np.ones_like(player_regrets)


	def _compute_ranges(self):
		''' Using the players' current strategies, computes their
			probabilities of reaching each state of the lookahead.
		'''
		PC, CC = constants.players_count, game_settings.card_count
		for d in range(1, self.depth):
			current_level_ranges = self.ranges_data[d]
			next_level_ranges = self.ranges_data[d+1]

			prev_layer_terminal_actions_count = self.terminal_actions_count[d-1]
			prev_layer_actions_count = self.actions_count[d-1] # ? - in lua 1-1 = 0 idx ?
			prev_layer_bets_count = self.bets_count[d-1]
			gp_layer_nonallin_bets_count = self.nonallinbets_count[d-2]
			gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
			# ? - indexing here :/
			# copy the ranges of inner nodes and transpose (np.transpose - swaps axis: 1dim <-> 2 dim)
			self.inner_nodes[d] = np.transpose(current_level_ranges[ prev_layer_terminal_actions_count: , :gp_layer_nonallin_bets_count , : , : , : ], [0,2,1,3,4])
			super_view = self.inner_nodes[d]
			super_view = super_view.reshape([1, prev_layer_bets_count, -1, PC, CC])
			super_view = super_view * np.ones_like(next_level_ranges)
			next_level_strategies = self.current_strategy_data[d+1]
			next_level_ranges = super_view.copy()
			# multiply the ranges of the acting player by his strategy
			next_level_ranges[ : , : , : , self.acting_player[d], : ] *= next_level_strategies


	def _compute_update_average_strategies(self, iter):
		''' Updates the players' average strategies with their current strategies.
		@param: iter the current iteration number of re-solving
		'''
		if iter > arguments.cfr_skip_iters:
			# no need to go through layers since we care for the average strategy only in the first node anyway
			# note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
			self.average_strategies_data[2] += self.current_strategy_data[2]


	def _compute_terminal_equities_terminal_equity(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at each lookahead state which is a terminal state of the game.
		'''
		CC = game_settings.card_count
		for d in range(2, self.depth+1):
			# call term eq evaluation
			if self.tree.street == 1:
				if d > 2 or self.first_call_terminal:
					ranges = self.ranges_data[d][1][-1].reshape([-1,CC])
					self.terminal_equity.call_value(ranges, self.cfvs_data[d][1][-1].reshape([-1,CC]))
			else:
				assert(self.tree.street == 2)
				# on river, any call is terminal
				if d > 2 or self.first_call_terminal:
					ranges = self.ranges_data[d][1].reshape([-1,CC])
					self.terminal_equity.call_value(ranges, self.cfvs_data[d][1].reshape([-1,CC]))
			# folds
			ranges = self.ranges_data[d][0].reshape([-1,CC])
			self.terminal_equity.fold_value(ranges, self.cfvs_data[d][0].reshape([-1,CC]))
			# correctly set the folded player by mutliplying by -1
			# fold_mutliplier = self.acting_player[d]*2 - 3
			fold_mutliplier = -1 if self.acting_player[d] == constants.players.P1 else 1
			self.cfvs_data[d][ 0, : , : , 0, : ] *= fold_mutliplier)
			self.cfvs_data[d][ 0, : , : , 1, : ] *= -fold_mutliplier)


	def _compute_terminal_equities_next_street_box(self):
		''' Using the players' reach probabilities, calls the neural net to
			compute the players' counterfactual values at the depth-limited
			states of the lookahead.
		'''
		PC, CC = constants.players_count, game_settings.card_count
		assert(self.tree.street == 1)
		for d in range(2, self.depth+1):
			if d > 2 or self.first_call_transition:
				self.next_street_boxes_inputs = self.next_street_boxes_inputs or {}
				self.next_street_boxes_outputs = self.next_street_boxes_outputs or {}

				self.next_street_boxes_inputs[d] = self.next_street_boxes_inputs[d] or np.zeros_like(self.ranges_data[d][ 1, : , : , : , : ].reshape([-1,PC,CC]) # ? - indexe nera {1} -> keepdims?
				self.next_street_boxes_outputs[d] = self.next_street_boxes_outputs[d] or self.next_street_boxes_inputs[d].copy()

				# now the neural net accepts the input for P1 and P2 respectively, so we need to swap the ranges if necessary
				self.next_street_boxes_outputs[d] = self.ranges_data[d][ 1, : , : , : , : ].copy()
				if self.tree.current_player == constants.players.P1: # ? - buvo == 1
					self.next_street_boxes_inputs[d] = self.next_street_boxes_outputs[d].copy()
				else:
					self.next_street_boxes_inputs[d][ : , 0, : ] = self.next_street_boxes_outputs[d][ : , 1, : ].copy()
					self.next_street_boxes_inputs[d][ : , 1, : ] = self.next_street_boxes_outputs[d][ : , 0, : ].copy()
				self.next_street_boxes[d].get_value(self.next_street_boxes_inputs[d], self.next_street_boxes_outputs[d])
				# now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
				if self.tree.current_player == constants.players.P1: # ? - buvo == 1
					self.next_street_boxes_inputs[d] = self.next_street_boxes_outputs[d].copy()
					self.next_street_boxes_outputs[d][ : , 0, : ] = self.next_street_boxes_inputs[d][ : , 1, : ].copy()
					self.next_street_boxes_outputs[d][ : , 1, : ] = self.next_street_boxes_inputs[d][ : , 0, : ].copy()
				self.cfvs_data[d][ 1, : , : , : , : ] = self.next_street_boxes_outputs[d].copy()


	def get_chance_action_cfv(self, action_index, board):
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
		''' # ? - a lot of indexes (batch_idx can be -1)
		assert( not (action_index == 2 and self.first_call_terminal) )
		# check if we should not use the first layer for transition call
		if action_index == 1 and self.first_call_transition:
			box_outputs = np.zeros_like(self.next_street_boxes_inputs[2])
			assert(box_outputs.shape[0] == 1)
			batch_index = 0
			next_street_box = self.next_street_boxes[2]
			pot_mult = self.pot_size[2][2]
		else:
			batch_index = action_index - 1 # remove fold
			if self.first_call_transition:
				batch_index = batch_index - 1
			box_outputs = np.zeros_like(self.next_street_boxes_inputs[3])
			next_street_box = self.next_street_boxes[3]
			pot_mult = self.pot_size[3][2]
		next_street_box.get_value_on_board(board, box_outputs)
		box_outputs *= pot_mult
		local out = box_outputs[batch_index][3-self.tree.current_player]
		return out


	def _compute_terminal_equities(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at all terminal states of the lookahead.
			These include terminal states of the game and depth-limited states.
		'''
		if self.tree.street == 1:
			self._compute_terminal_equities_next_street_box()
		self._compute_terminal_equities_terminal_equity()
		# multiply by pot scale factor
		for d in range(2, self.depth+1):
			self.cfvs_data[d] *= self.pot_size[d]


	def _compute_cfvs(self):
		''' Using the players' reach probabilities and terminal counterfactual
			values, computes their cfvs at all states of the lookahead.
		'''
		for d in range(self.depth, 1, -1):
			gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
			ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]

			self.cfvs_data[d][ : , : , : , 0, : ] *= self.empty_action_mask[d] # ? - {1} buvo
			self.cfvs_data[d][ : , : , : , 1, : ] *= self.empty_action_mask[d] # ? - {2} buvo
			self.placeholder_data[d] = self.cfvs_data[d].copy()
			# player indexing is swapped for cfvs
			self.placeholder_data[d][ : , : , : , self.acting_player[d], : ] *= self.current_strategy_data[d]
			self.regrets_sum[d] = np.sum(self.placeholder_data[d], axis=0, keepdims=True)
			# use a swap placeholder to change [[1,2,3], [4,5,6]] into [[1,2], [3,4], [5,6]]
			swap = self.swap_data[d-1]
			swap = self.regrets_sum[d].copy()
			self.cfvs_data[d-1][ gp_layer_terminal_actions_count: , :ggp_layer_nonallin_bets_count , : , : , : ] = np.transpose(swap, [0,2,1,3,4]).copy() # ? - transpose(2,3))


	def _compute_cumulate_average_cfvs(self, iter):
		''' Updates the players' average counterfactual values with their
			cfvs from the current iteration.
		@param: iter the current iteration number of re-solving
		'''
		if iter > arguments.cfr_skip_iters:
		    self.average_cfvs_data[1] += self.cfvs_data[1]
		    self.average_cfvs_data[2] += self.cfvs_data[2]


	def _compute_normalize_average_strategies(self):
		''' Normalizes the players' average strategies.
			Used at the end of re-solving so that we can track
			un-normalized average strategies, which are simpler to compute.
		'''
		# using regrets_sum as a placeholder container
		player_avg_strategy = self.average_strategies_data[2]
		player_avg_strategy_sum = self.regrets_sum[2]
		player_avg_strategy_sum = np.sum(player_avg_strategy, axis=0, keepdims=True)
		player_avg_strategy -= player_avg_strategy_sum * np.ones_like(player_avg_strategy)
		# if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
		# it sums to one -> now we set to always fold
		player_avg_strategy[1][ player_avg_strategy[1] != player_avg_strategy[1] ] = 1
		player_avg_strategy[ player_avg_strategy != player_avg_strategy ] = 0


	def _compute_normalize_average_cfvs(self):
		''' Normalizes the players' average counterfactual values.
			Used at the end of re-solving so that we can track
			un-normalized average cfvs, which are simpler to compute.
		'''
		self.average_cfvs_data[1] /= (arguments.cfr_iters - arguments.cfr_skip_iters)


	def _compute_regrets(self):
		''' Using the players' counterfactual values, updates their
			total regrets for every state in the lookahead.
		'''
		CC = game_settings.card_count
		for d in range(self.depth, 1, -1):
			gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
			gp_layer_bets_count = self.bets_count[d-2]
			ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]
			current_regrets = self.current_regrets_data[d]
			current_regrets = self.cfvs_data[d][ : , : , : , self.acting_player[d], : ].copy()
			next_level_cfvs = self.cfvs_data[d-1]
			parent_inner_nodes = self.inner_nodes_p1[d-1]
			parent_inner_nodes = np.transpose(next_level_cfvs[ gp_layer_terminal_actions_count: , :ggp_layer_nonallin_bets_count, : , self.acting_player[d], : ], [0,2,1,3,4]) # :transpose(2,3))
			parent_inner_nodes = parent_inner_nodes.reshape([1, gp_layer_bets_count, -1, CC])
			parent_inner_nodes = parent_inner_nodes * np.ones_like(current_regrets)
			current_regrets -= parent_inner_nodes
			self.regrets_data[d] += current_regrets
			# (CFR+)
			self.regrets_data[d] = np.clip(self.regrets_data[d], 0, constants.max_number)


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
		actions_count = self.average_strategies_data[2].shape[0]
		PC, CC, AC = constants.players_count, game_settings.card_count, actions_count
		out = {}
		# 1.0 average strategy
		# [actions x range]
		# lookahead already computes the averate strategy we just convert the dimensions
		out.strategy = self.average_strategies_data[2].reshape([-1,CC]).copy()
		# 2.0 achieved opponent's CFVs at the starting node
		out.achieved_cfvs = self.average_cfvs_data[1].reshape([PC,CC])[0].copy()
		# 3.0 CFVs for the acting player only when resolving first node
		if self.reconstruction_opponent_cfvs:
			out.root_cfvs = None
		else:
			out.root_cfvs = self.average_cfvs_data[1].reshape([PC,CC])[1].copy()
			# swap cfvs indexing
			out.root_cfvs_both_players = self.average_cfvs_data[1].reshape([PC,CC]).copy()
			out.root_cfvs_both_players[2] = self.average_cfvs_data[1].reshape([PC,CC])[1].copy()
			out.root_cfvs_both_players[1] = self.average_cfvs_data[1].reshape([PC,CC])[2].copy()
		# 4.0 children CFVs
		# [actions x range]
		out.children_cfvs = self.average_cfvs_data[2][ : , : , : , 1, : ].copy().reshape([-1,CC])
		# IMPORTANT divide average CFVs by average strategy in here
		scaler = self.average_strategies_data[2].reshape([-1,CC]).copy()
		range_mul = self.ranges_data[1][ : , : , : , 1, : ].reshape([1,CC]).copy()
		range_mul = range_mul * np.ones_like(scaler)
		scaler = scaler * range_mul
		scaler = np.sum(scaler, axis=1, keepdims=True) * np.ones_like(range_mul)
		scaler = scaler * (arguments.cfr_iters - arguments.cfr_skip_iters)
		out.children_cfvs /= scaler
		assert(out.strategy)
		assert(out.achieved_cfvs)
		assert(out.children_cfvs)
		return out


	def _set_opponent_starting_range(self, iteration):
		''' Generates the opponent's range for the current re-solve iteration
			using the @{cfrd_gadget|CFRDGadget}.
		@param: iteration the current iteration number of re-solving
		'''
		if self.reconstruction_opponent_cfvs:
			# note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
			opponent_range = self.reconstruction_gadget.compute_opponent_range(self.cfvs_data[1][ : , : , : , 0, : ], iteration)
			self.ranges_data[1][ : , : , : , 1, : ] = opponent_range.copy()




#
