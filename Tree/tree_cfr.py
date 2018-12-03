'''
	Runs Counterfactual Regret Minimization (CFR) to approximately
	solve a game represented by a complete game tree.
	As this class does full solving from the root of the game with no
	limited lookahead, it is not used in continual re-solving.
	It is provided simply for convenience.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from Game.card_tools import card_tools
from Game.card_to_string_conversion import card_to_string
from TerminalEquity.terminal_equity import TerminalEquity

class TreeCFR():
	def __init__(self):
		''' for ease of implementation, we use small epsilon rather than
			zero when working with regrets
		'''
		self.regret_epsilon = 1/1000000000
		self._cached_terminal_equities = {}


	def _get_terminal_equity(self, node):
		''' Gets an evaluator for player equities at a terminal node.
			Caches the result to minimize creation of
			@{terminal_equity|TerminalEquity} objects.
		@param: node the terminal node to evaluate
		@return a @{terminal_equity|TerminalEquity} evaluator for the node
		'''
		# board_idx = card_to_string.cards_to_string(node.board)
		if node.board.ndim == 0: board_idx = -2
		else: board_idx = card_tools.get_board_index(node.board)
		try:
			cached = self._cached_terminal_equities[board_idx]
		except:
			cached = TerminalEquity()
			cached.set_board(node.board)
			self._cached_terminal_equities[board_idx] = cached
		return cached


	def cfrs_iter_dfs(self, node, iter):
		''' Recursively walks the tree, applying the CFR algorithm.
		@param: node the current node in the tree
		@param: iter the current iteration number
		'''
		assert (node.current_player == constants.players.P1 or\
				node.current_player == constants.players.P2 or\
				node.current_player == constants.players.chance)
		actions_count = len(node.children)
		AC, HC, PC = actions_count, game_settings.hand_count, constants.players_count
		opponent_index = 1 - node.current_player
		# dimensions in tensor
		action_dim = 0
		card_dim = 1
		# compute values using terminal_equity in terminal nodes
		if node.terminal:
			terminal_equity = self._get_terminal_equity(node)
			values = np.zeros_like(node.ranges_absolute)
			if node.type == constants.node_types.terminal_fold:
				terminal_equity.tree_node_fold_value(node.ranges_absolute, values, opponent_index)
			else:
				terminal_equity.tree_node_call_value(node.ranges_absolute, values)
			values *= node.pot # multiply by the pot
			node.cf_values = values.reshape(node.ranges_absolute.shape)
		else:
			if node.current_player == constants.players.chance:
				current_strategy = node.strategy
			else: # we have to compute current strategy at the beginning of each iteraton
				# initialize regrets in the first iteration
				if node.regrets is None: node.regrets = np.full([AC,HC], self.regret_epsilon, dtype=arguments.dtype)
				if node.possitive_regrets is None: node.possitive_regrets = np.full([AC,HC], self.regret_epsilon, dtype=arguments.dtype)
				# compute positive regrets so that we can compute the current strategy from them
				node.possitive_regrets = node.regrets.copy()
				node.possitive_regrets[node.possitive_regrets <= self.regret_epsilon] = self.regret_epsilon
				# compute the current strategy
				regrets_sum = node.possitive_regrets.sum(axis=action_dim, keepdims=True)
				current_strategy = node.possitive_regrets.copy()
				current_strategy /= ( regrets_sum * np.ones_like(current_strategy) )
			# current cfv [[actions, players, ranges]]
			cf_values_allactions = np.zeros([AC,PC,HC], dtype=arguments.dtype)
			children_ranges_absolute = {}
			if node.current_player == constants.players.chance:
				ranges_mul_matrix = node.ranges_absolute[0] * np.ones([AC,1], dtype=node.ranges_absolute.dtype) # ?
				children_ranges_absolute[0] = current_strategy * ranges_mul_matrix # ?
				ranges_mul_matrix = node.ranges_absolute[1] * np.ones([AC,1], dtype=node.ranges_absolute.dtype) # ?
				children_ranges_absolute[1] = current_strategy * ranges_mul_matrix # ?
			else:
				ranges_mul_matrix = node.ranges_absolute[node.current_player] * np.ones([AC,1], dtype=node.ranges_absolute.dtype)
				children_ranges_absolute[node.current_player] = current_strategy * ranges_mul_matrix # ?
				children_ranges_absolute[opponent_index] = node.ranges_absolute[opponent_index] * np.ones([AC,1], dtype=node.ranges_absolute.dtype)
			for i in range(len(node.children)):
				child_node = node.children[i]
				# set new absolute ranges (after the action) for the child
				child_node.ranges_absolute = node.ranges_absolute.copy()
				child_node.ranges_absolute[0] = children_ranges_absolute[0][i].copy()
				child_node.ranges_absolute[1] = children_ranges_absolute[1][i].copy() # == :copy() ?
				self.cfrs_iter_dfs(child_node, iter)
				cf_values_allactions[i] = child_node.cf_values
			node.cf_values = np.zeros([PC,HC], dtype=arguments.dtype)
			if node.current_player != constants.players.chance:
				strategy_mul_matrix = current_strategy.reshape([AC,HC])
				node.cf_values[node.current_player] = (strategy_mul_matrix * cf_values_allactions[ : ,node.current_player, : ]).sum(axis=0, keepdims=True) # ?
				node.cf_values[opponent_index] = cf_values_allactions[ : , opponent_index, : ].sum(axis=0, keepdims=True) # ?
			else:
				node.cf_values[0] = cf_values_allactions[ : , 0, : ].sum(axis=0, keepdims=True) # ?
				node.cf_values[1] = cf_values_allactions[ : , 1, : ].sum(axis=0, keepdims=True) # ?
			if node.current_player != constants.players.chance:
				# computing regrets
				current_regrets = cf_values_allactions[ : , node.current_player, : ][ : , np.newaxis, : ].reshape([AC,HC]).copy() # ?
				current_regrets -= node.cf_values[node.current_player].reshape([1,HC]) * np.ones_like(current_regrets)
				self.update_regrets(node, current_regrets)
				# aHCumulating average strategy
				self.update_average_strategy(node, current_strategy, iter)


	def update_regrets(self, node, current_regrets):
		''' Update a node's total regrets with the current iteration regrets.
		@param: node the node to update
		@param: current_regrets the regrets from the current iteration of CFR
		'''
		node.regrets += current_regrets
		node.regrets[ node.regrets <= self.regret_epsilon ] = self.regret_epsilon


	def update_average_strategy(self, node, current_strategy, iter):
		''' Update a node's average strategy with the current iteration strategy.
		@param: node the node to update
		@param: current_strategy the CFR strategy for the current iteration
		@param: iter the iteration number of the current CFR iteration
		'''
		actions_count = len(node.children)
		AC, HC = actions_count, game_settings.hand_count
		if iter > self.cfr_skip_iters:
			node.strategy = np.zeros([AC,HC], dtype=arguments.dtype) if node.strategy is None else node.strategy
			node.iter_weight_sum = np.zeros([HC], dtype=arguments.dtype) if node.iter_weight_sum is None else node.iter_weight_sum
			iter_weight_contribution = node.ranges_absolute[node.current_player].copy()
			iter_weight_contribution[iter_weight_contribution <= 0] = self.regret_epsilon
			node.iter_weight_sum += iter_weight_contribution
			iter_weight = iter_weight_contribution / node.iter_weight_sum
			expanded_weight = iter_weight.reshape([1,HC]) * np.ones_like(node.strategy)
			old_strategy_scale = expanded_weight * (-1) + 1 # same as 1 - expanded weight
			node.strategy *= old_strategy_scale
			strategy_addition = current_strategy * expanded_weight
			node.strategy += strategy_addition


	def run_cfr(self, root, starting_ranges, iter_count=arguments.cfr_iters, skip=arguments.cfr_skip_iters):
		''' Run CFR to solve the given game tree.
		@param root the root node of the tree to solve.
		@param: [opt] starting_ranges probability vectors over player private hands
				at the root node (default uniform)
		@param: [opt] iter_count the number of iterations to run CFR for
				(default @{arguments.cfr_iters})
		'''
		assert(starting_ranges is not None)
		root.ranges_absolute = starting_ranges
		self.cfr_skip_iters = skip
		for self.iter in range(iter_count):
			self.cfrs_iter_dfs(root, self.iter)



#
