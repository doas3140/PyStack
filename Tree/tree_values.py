'''
	Computes the expected value of a strategy profile on a game's public tree,
	as well as the value of a best response against the profile.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from Game.card_tools import card_tools
from TerminalEquity.terminal_equity import TerminalEquity

class TreeValues():
	def __init__(self):
		self.terminal_equity = TerminalEquity()


	def _fill_ranges_dfs(self, node, ranges_absolute):
		''' Recursively walk the tree and calculate the probability
			of reaching each node using the saved strategy profile.
			The reach probabilities are saved in the
			`ranges_absolute` field of each node.
		@param: node the current node of the tree
		@param: ranges_absolute a (2,K) tensor containing the probabilities of each
				player reaching the current node with each private hand
		'''
		node.ranges_absolute = ranges_absolute.copy()
		if node.terminal:
			return
		assert(node.strategy is not None)
		actions_count = len(node.children)
		AC = actions_count
		# check that it's a legal strategy
		strategy_to_check = node.strategy
		hands_mask = card_tools.get_possible_hand_indexes(node.board)
		if node.current_player != constants.players.chance:
			checksum = strategy_to_check.sum(axis=0)
			assert(not np.any(strategy_to_check < 0))
			assert(not np.any(checksum > 1.001))
			assert(not np.any(checksum < 0.999))
			assert(not np.any(checksum != checksum))
		assert((node.ranges_absolute < 0).sum() == 0)
		assert((node.ranges_absolute > 1).sum() == 0)
		# check if the range consists only of cards that don't overlap with the board
		CC = game_settings.card_count
		PC = constants.players_count
		impossible_hands_mask = np.ones_like(hands_mask,dtype=int) - hands_mask
		impossible_range_sum = (node.ranges_absolute.copy() * impossible_hands_mask.reshape([1,CC])).sum() # ? delete .copy()
		assert(impossible_range_sum == 0, impossible_range_sum)
		children_ranges_absolute = np.zeros([len(node.children), PC, CC], dtype=float)
		# chance player
		if node.current_player == constants.players.chance:
			# multiply ranges of both players by the chance prob
			children_ranges_absolute[ : , constants.players.P1, : ] = node.ranges_absolute[constants.players.P1] * np.ones([AC,1], dtype=children_ranges_absolute.dtype)
			children_ranges_absolute[ : , constants.players.P2, : ] = node.ranges_absolute[constants.players.P2] * np.ones([AC,1], dtype=children_ranges_absolute.dtype)

			children_ranges_absolute[ : , constants.players.P1, : ] *= node.strategy
			children_ranges_absolute[ : , constants.players.P2, : ] *= node.strategy
		else: # player
			# copy the range for the non-acting player
			opponent = 1 - node.current_player
			children_ranges_absolute[ : , opponent, : ] = node.ranges_absolute[opponent].copy() * np.ones([AC,1], dtype=children_ranges_absolute.dtype)
			# multiply the range for the acting player using his strategy
			current_range_matrix = node.ranges_absolute[node.current_player] * np.ones([AC,1], dtype=node.ranges_absolute.dtype)
			children_ranges_absolute[ : , node.current_player, : ] = node.strategy * current_range_matrix
		# fill the ranges for the children
		for i in range(len(node.children)):
			child_node = node.children[i]
			child_range = children_ranges_absolute[i]
			# go deeper
			self._fill_ranges_dfs(child_node, child_range)


	def _compute_values_dfs(self, node):
		''' Recursively calculate the counterfactual values for each player
			at each node of the tree using the saved strategy profile.
			The cfvs for each player in the given strategy profile when playing
			against each other is stored in the `cf_values` field for each node.
			The cfvs for a best response against each player in the profile are
			stored in the `cf_values_br` field for each node.
		@param: node the current node
		'''
		# compute values using terminal_equity in terminal nodes
		if node.terminal:
			assert (node.type == constants.node_types.terminal_fold or\
					node.type == constants.node_types.terminal_call)
			self.terminal_equity.set_board(node.board)
			values = np.zeros_like(node.ranges_absolute)
			if node.type == constants.node_types.terminal_fold:
				opponent = 1 - node.current_player
				self.terminal_equity.tree_node_fold_value(node.ranges_absolute, values, folding_player=opponent)
			else:
				self.terminal_equity.tree_node_call_value(node.ranges_absolute, values)
			# multiply by the pot
			values *= node.pot
			node.cf_values = values.reshape(node.ranges_absolute.shape)
			node.cf_values_br = values.reshape(node.ranges_absolute.shape)
		else:
			actions_count = len(node.children)
			AC = actions_count
			ranges_size = node.ranges_absolute.shape[1]
			# [[actions, players, ranges]]
			cf_values_allactions = np.zeros([len(node.children), 2, ranges_size], dtype=float)
			cf_values_br_allactions = np.zeros([len(node.children), 2, ranges_size], dtype=float)
			for i in range(len(node.children)):
				child_node = node.children[i]
				self._compute_values_dfs(child_node)
				cf_values_allactions[i] = child_node.cf_values
				cf_values_br_allactions[i] = child_node.cf_values_br
			node.cf_values = np.zeros([2, ranges_size], dtype=float)
			node.cf_values_br = np.zeros([2, ranges_size], dtype=float)
			# strategy = [[actions x range]]
			strategy_mul_matrix = node.strategy.reshape([AC, ranges_size])
			# compute CFVs given the current strategy for this node
			if node.current_player == constants.players.chance:
				node.cf_values = cf_values_allactions.sum(axis=0)
				node.cf_values_br = cf_values_br_allactions.sum(axis=0)
			else:
				opponent = 1 - node.current_player
				node.cf_values[node.current_player] = (strategy_mul_matrix * cf_values_allactions[ : , node.current_player, : ]).sum(axis=0, keepdims=True)
				node.cf_values[opponent] = cf_values_allactions[ : , opponent, : ].sum(axis=0, keepdims=True)
				# compute CFVs given the BR strategy for this node
				node.cf_values_br[opponent] = cf_values_br_allactions[ : , opponent, : ].sum(axis=0, keepdims=True)
				node.cf_values_br[node.current_player] = cf_values_br_allactions[ : , node.current_player, : ].max(axis=0, keepdims=True)
		# counterfactual values weighted by the reach prob
		node.cfv_infset = np.zeros([2])
		node.cfv_infset[0] = np.dot(node.cf_values[0], node.ranges_absolute[0])
		node.cfv_infset[1] = np.dot(node.cf_values[1], node.ranges_absolute[1])
		# compute CFV-BR values weighted by the reach prob
		node.cfv_br_infset = np.zeros([2])
		node.cfv_br_infset[0] = np.dot(node.cf_values_br[0], node.ranges_absolute[0])
		node.cfv_br_infset[1] = np.dot(node.cf_values_br[1], node.ranges_absolute[1])
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
		@param: root The root of the game tree. Each node of the tree is assumed
				to have a strategy saved in the `strategy` field.
		@param: [opt] starting_ranges probability vectors over player private hands
				at the root node (default uniform)
		'''
		PC, CC = constants.players_count, game_settings.card_count
		# 1.0 set the starting range
		uniform_ranges = np.full([PC,CC], 1/CC, dtype=float)
		starting_ranges = starting_ranges or uniform_ranges
		# 2.0 check the starting ranges
		checksum = starting_ranges.sum(axis=1)
		assert(abs(checksum[0] - 1) < 0.0001, 'starting range does not sum to 1')
		assert(abs(checksum[1] - 1) < 0.0001, 'starting range does not sum to 1')
		assert((starting_ranges < 0).sum() == 0)
		# 3.0 compute the values
		self._fill_ranges_dfs(root, starting_ranges)
		self._compute_values_dfs(root)




#
