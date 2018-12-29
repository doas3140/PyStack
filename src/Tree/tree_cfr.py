'''
	Runs Counterfactual Regret Minimization (CFR) to approximately
	solve a game represented by a complete game tree.
	This class does full solving from the root of the game with no
	limited lookahead, it is not used in continual re-solving.
	It is provided simply for convenience.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from Game.card_to_string_conversion import card_to_string
from TerminalEquity.terminal_equity import TerminalEquity

class TreeCFR():
	def __init__(self):
		self._cached_terminal_equities = {}

	def _get_board_index(self, board):
		''' Gives a numerical index for a set of board cards.
			(used only in self._get_terminal_equity)
		@param: [3-5] :board a non-empty vector of cards
		@return int   :the numerical index for the board
		'''
		CC = constants.card_count
		used_cards = np.zeros([CC], dtype=arguments.dtype)
		for i in range(board.shape[0] - 1):
			used_cards[ board[i] ] = 1
		ans = -1
		for i in range(CC):
			if used_cards[i] == 0:
				ans += 1
			if i == board[-1]:
				return ans
		return -1

	def _get_terminal_equity(self, node):
		''' Gets an evaluator for player equities at a terminal node.
			Caches the result to minimize creation time of objects.
		@param: Node           :the terminal node to evaluate
		@return TerminalEquity :evaluator for the node
		'''
		# board_idx = card_to_string.cards_to_string(node.board)
		if node.board.ndim == 0: board_idx = -2
		else: board_idx = self._get_board_index(node.board)
		try:
			cached = self._cached_terminal_equities[board_idx]
		except:
			cached = TerminalEquity()
			cached.set_board(node.board)
			self._cached_terminal_equities[board_idx] = cached
		return cached


	def cfrs_iter_dfs(self, node, iter):
		''' Recursively walks the tree, applying the CFR algorithm
		@param: Node :node the current node in the tree
		@param: Int  :iter the current iteration number (used in self.update_average_strategy)
		'''
		actions_count = len(node.children)
		AC, HC, PC = actions_count, constants.hand_count, constants.players_count
		opponent_index = 1 - node.current_player
		# compute values using terminal_equity in terminal nodes
		if node.terminal:
			terminal_equity = self._get_terminal_equity(node)
			values = np.zeros_like(node.ranges)
			if node.type == constants.node_types.terminal_fold:
				fold_matrix = terminal_equity.get_fold_matrix()
				result[ 0 , : ] = np.dot(node.ranges[1], fold_matrix)
				result[ 1 , : ] = np.dot(node.ranges[0], fold_matrix)
				result[ opponent_index ] *= -1
			else:
				equity_matrix = terminal_equity.get_equity_matrix()
				values[ 0 , : ] = np.dot(node.ranges[1], equity_matrix)
				values[ 1 , : ] = np.dot(node.ranges[0], equity_matrix)
			values *= node.pot # multiply by the pot
			node.cf_values = values.reshape(node.ranges.shape)
		else:
			if node.current_player == constants.players.chance:
				current_strategy = node.strategy
			else: # we have to compute current strategy at the beginning of each iteraton
				# initialize regrets in the first iteration
				if node.regrets is None: node.regrets = np.full([AC,HC], constants.regret_epsilon, dtype=arguments.dtype)
				if node.possitive_regrets is None: node.possitive_regrets = np.full([AC,HC], constants.regret_epsilon, dtype=arguments.dtype)
				# compute positive regrets so that we can compute the current strategy from them
				node.possitive_regrets = node.regrets.copy()
				node.possitive_regrets[node.possitive_regrets <= constants.regret_epsilon] = constants.regret_epsilon
				# compute the current strategy
				regrets_sum = node.possitive_regrets.sum(axis=0, keepdims=True)
				current_strategy = node.possitive_regrets.copy()
				current_strategy /= regrets_sum
			# current cfv [[actions, players, ranges]]
			cf_values_allactions = np.zeros([AC,PC,HC], dtype=arguments.dtype)
			children_ranges = {}
			if node.current_player == constants.players.chance:
				children_ranges[0] = current_strategy * node.ranges[0].reshape([1,-1])
				children_ranges[1] = current_strategy * node.ranges[1].reshape([1,-1])
			else:
				children_ranges[node.current_player] = current_strategy * node.ranges[node.current_player].reshape([1,-1])
				children_ranges[opponent_index] = np.repeat(node.ranges[opponent_index].reshape([1,-1]), AC, axis=0)
			for i in range(len(node.children)):
				child_node = node.children[i]
				# set new absolute ranges (after the action) for the child
				child_node.ranges = node.ranges.copy()
				child_node.ranges[0] = children_ranges[0][i]
				child_node.ranges[1] = children_ranges[1][i]
				self.cfrs_iter_dfs(child_node, iter)
				cf_values_allactions[i] = child_node.cf_values
			node.cf_values = np.zeros([PC,HC], dtype=arguments.dtype)
			if node.current_player != constants.players.chance:
				strategy_mul_matrix = current_strategy.reshape([AC,HC])
				node.cf_values[node.current_player] = (strategy_mul_matrix * cf_values_allactions[ : ,node.current_player, : ]).sum(axis=0, keepdims=True)
				node.cf_values[opponent_index] = cf_values_allactions[ : , opponent_index, : ].sum(axis=0, keepdims=True)
			else:
				node.cf_values[0] = cf_values_allactions[ : , 0, : ].sum(axis=0, keepdims=True)
				node.cf_values[1] = cf_values_allactions[ : , 1, : ].sum(axis=0, keepdims=True)
			if node.current_player != constants.players.chance:
				# computing regrets
				current_regrets = cf_values_allactions[ : , node.current_player, : ]
				current_regrets -= node.cf_values[node.current_player].reshape([1,HC])
				# updating regrets
				node.regrets += current_regrets
				node.regrets[ node.regrets <= constants.regret_epsilon ] = constants.regret_epsilon
				# acumulating average strategy
				if iter > self.cfr_skip_iters:
					self.update_average_strategy(node, current_strategy)


	def update_average_strategy(self, node, current_strategy):
		''' Update a node's average strategy with the current iteration strategy.
		@param: Node :node the node to update
		@param: [I]  :current_strategy the CFR strategy for the current iteration
		'''
		actions_count = len(node.children)
		AC, HC = actions_count, constants.hand_count
		node.strategy = np.zeros([AC,HC], dtype=arguments.dtype) if node.strategy is None else node.strategy
		node.iter_weight_sum = np.zeros([HC], dtype=arguments.dtype) if node.iter_weight_sum is None else node.iter_weight_sum
		iter_weight_contribution = node.ranges[node.current_player].copy()
		iter_weight_contribution[iter_weight_contribution <= 0] = constants.regret_epsilon
		node.iter_weight_sum += iter_weight_contribution
		iter_weight = iter_weight_contribution / node.iter_weight_sum
		expanded_weight = np.repeat(iter_weight.reshape([1,HC]), AC, axis=0)
		old_strategy_scale = expanded_weight * (-1) + 1 # same as 1 - expanded weight
		node.strategy *= old_strategy_scale
		strategy_addition = current_strategy * expanded_weight
		node.strategy += strategy_addition


	def run_cfr(self, root, starting_ranges, iter_count=arguments.cfr_iters, skip=arguments.cfr_skip_iters):
		''' Run CFR to solve the given game tree
		@param: Node  :root the root node of the tree to solve
		@param: [P,I] :probability vectors over player private hands at the root node
		@param: int   :number of iterations to run CFR for
		@param: int   :number of iterations to emit when calculating average strategy
		'''
		root.ranges = starting_ranges
		self.cfr_skip_iters = skip
		for self.iter in range(iter_count):
			self.cfrs_iter_dfs(root, self.iter)



#
