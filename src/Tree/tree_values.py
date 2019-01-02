'''
	Computes the expected value of a strategy profile on a game's public tree,
	as well as the value of a best response against the profile
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from TerminalEquity.terminal_equity import TerminalEquity

class TreeValues():
	def __init__(self):
		self.terminal_equity = TerminalEquity()


	def _fill_ranges_dfs(self, node, ranges):
		''' Recursively walk the tree and calculate the probability
			of reaching each node using the saved strategy profile.
			The reach probabilities are saved in the `ranges` field of each node
		@param: Node  :current node of the tree
		@param: [P,I] :tensor containing the probabilities of each player
					   reaching the current node with each private hand
		'''
		actions_count = len(node.children)
		HC, PC, AC = constants.hand_count, constants.players_count, actions_count
		P1, P2 = constants.players.P1, constants.players.P2
		node.ranges = ranges.copy()
		if node.terminal:
			return
		assert(node.strategy is not None)
		# check that it's a legal strategy
		if node.current_player != constants.players.chance:
			node.strategy[ node.strategy < 0 ] = 0 # kartais skaiciai yra -1e11
			checksum = node.strategy.sum(axis=0)
			assert(not np.any(node.strategy < 0))
			assert(not np.any(checksum > 1.001))
			assert(not np.any(checksum < 0.999))
			assert(not np.any(checksum != checksum))
		assert((node.ranges < 0).sum() == 0)
		assert((node.ranges > 1).sum() == 0)
		# check if the range consists only of cards that don't overlap with the board
		hands_mask = card_tools.get_possible_hands_mask(node.board)
		impossible_hands_mask = np.ones_like(hands_mask,dtype=arguments.int_dtype) - hands_mask
		impossible_range_sum = (node.ranges * impossible_hands_mask.reshape([1,HC])).sum()
		assert(impossible_range_sum == 0, impossible_range_sum)
		children_ranges = np.zeros([len(node.children), PC, HC], dtype=arguments.dtype)
		# chance node
		if node.current_player == constants.players.chance:
			# multiply ranges of both players by the chance prob
			children_ranges[ : , P1 , : ] = node.ranges[P1].reshape([1,-1]) * node.strategy
			children_ranges[ : , P2 , : ] = node.ranges[P2].reshape([1,-1]) * node.strategy
		else: # player node
			# copy the range for the non-acting player (opponent)
			children_ranges[ : , 1 - node.current_player , : ] = np.repeat(node.ranges[opponent].reshape([1,-1]), AC, axis=0)
			# multiply the range for the acting player using his strategy
			children_ranges[ : , node.current_player , : ] = node.ranges[node.current_player].reshape([1,-1]) * node.strategy
		# fill the ranges for the children
		for i in range(len(node.children)):
			# go deeper
			self._fill_ranges_dfs(node.children[i], children_ranges[i])


	def _compute_values_dfs(self, node):
		''' Recursively calculate the counterfactual values for each player
			at each node of the tree using the saved strategy profile.
			The cfvs for each player in the given strategy profile when playing
			against each other is stored in the `cf_values` field for each node.
			The cfvs for a best response against each player in the profile are
			stored in the `cf_values_br` field for each node.
		@param: Node :current node
		'''
		# compute values using terminal_equity in terminal nodes
		if node.terminal:
			assert (node.type == constants.node_types.terminal_fold or node.type == constants.node_types.terminal_call)
			self.terminal_equity.set_board(node.board)
			values = np.zeros_like(node.ranges)
			if node.type == constants.node_types.terminal_fold:
				opponent = 1 - node.current_player
				fold_matrix = self.terminal_equity.get_fold_matrix()
				values[ 0 , : ] = np.dot(node.ranges[1], fold_matrix)
				values[ 1 , : ] = np.dot(node.ranges[0], fold_matrix)
				values[ opponent ] *= -1
			else:
				equity_matrix = self.terminal_equity.get_equity_matrix()
				values[ 0 , : ] = np.dot(node.ranges[1], equity_matrix)
				values[ 1 , : ] = np.dot(node.ranges[0], equity_matrix)
			# multiply by the pot
			values *= node.pot
			node.cf_values = node.cf_values_br = values
		else:
			actions_count = len(node.children)
			AC, HC = actions_count, constants.hand_count
			# [[actions, players, ranges]]
			cf_values_allactions = np.zeros([len(node.children), 2, HC], dtype=arguments.dtype)
			cf_values_br_allactions = np.zeros([len(node.children), 2, HC], dtype=arguments.dtype)
			for i in range(len(node.children)):
				child_node = node.children[i]
				self._compute_values_dfs(child_node)
				cf_values_allactions[i] = child_node.cf_values
				cf_values_br_allactions[i] = child_node.cf_values_br
			node.cf_values = np.zeros([2,HC], dtype=arguments.dtype)
			node.cf_values_br = np.zeros([2,HC], dtype=arguments.dtype)
			# strategy = [actions, range]
			strategy_mul_matrix = node.strategy.reshape([AC,HC])
			# compute CFVs given the current strategy for this node
			if node.current_player == constants.players.chance:
				node.cf_values = cf_values_allactions.sum(axis=0)
				node.cf_values_br = cf_values_br_allactions.sum(axis=0)
			else:
				current_player, opponent = node.current_player, 1 - node.current_player
				node.cf_values[current_player] = (node.strategy * cf_values_allactions[ : , current_player, : ]).sum(axis=0)
				node.cf_values[opponent] = cf_values_allactions[ : , opponent, : ].sum(axis=0)
				# compute CFVs given the BR strategy for this node
				node.cf_values_br[opponent] = cf_values_br_allactions[ : , opponent, : ].sum(axis=0)
				node.cf_values_br[current_player] = cf_values_br_allactions[ : , current_player, : ].max(axis=0)
		# counterfactual values weighted by the reach prob
		node.cfv_infset = np.zeros([2], dtype=arguments.dtype)
		node.cfv_infset[0] = np.dot(node.cf_values[0], node.ranges[0])
		node.cfv_infset[1] = np.dot(node.cf_values[1], node.ranges[1])
		# compute CFV-BR values weighted by the reach prob
		node.cfv_br_infset = np.zeros([2], dtype=arguments.dtype)
		node.cfv_br_infset[0] = np.dot(node.cf_values_br[0], node.ranges[0])
		node.cfv_br_infset[1] = np.dot(node.cf_values_br[1], node.ranges[1])
		#
		node.epsilon = node.cfv_br_infset - node.cfv_infset
		node.exploitability = node.epsilon.mean()


	def compute_values(self, root, starting_ranges=None):
		''' Compute the self play and best response values of a strategy profile
			on the given game tree.
			The cfvs for each player in the given strategy profile when playing
			against each other is stored in the `cf_values` field for each node.
			The cfvs for a best response against each player in the profile are
			stored in the `cf_values_br` field for each node.
		@param: Node  :root of the game tree. Each node of the tree is assumed
					   to have a strategy saved in the `strategy` field.
		@param: [P,I] :(optional) probability vectors over player private hands
								  at the root node (default uniform)
		'''
		PC, HC = constants.players_count, constants.hand_count
		# 1.0 set the starting range (uniform if ranges=None)
		if starting_ranges is None: starting_ranges = np.full([PC,HC], 1/HC, dtype=arguments.dtype)
		# 2.0 check the starting ranges
		checksum = starting_ranges.sum(axis=1)
		assert(abs(checksum[0] - 1) < 0.0001) # starting range does not sum to 1
		assert(abs(checksum[1] - 1) < 0.0001) # starting range does not sum to 1
		assert((starting_ranges < 0).sum() == 0)
		# 3.0 compute the values
		self._fill_ranges_dfs(root, starting_ranges)
		self._compute_values_dfs(root)




#
