'''
	Computes the expected value of a strategy profile on a game's public tree,
	as well as the value of a best response against the profile.
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..Game.card_tools import card_tools
from ..TerminalEquity.terminal_equity import TerminalEquity

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
		pass
		# check that it's a legal strategy
		# check if the range consists only of cards that don't overlap with the board
		# chance player
		# multiply ranges of both players by the chance prob
		# player
		# copy the range for the non-acting player
		# multiply the range for the acting player using his strategy
		# fill the ranges for the children
		# go deeper


	def _compute_values_dfs(self, node):
		''' Recursively calculate the counterfactual values for each player
			at each node of the tree using the saved strategy profile.
			The cfvs for each player in the given strategy profile when playing
			against each other is stored in the `cf_values` field for each node.
			The cfvs for a best response against each player in the profile are
			stored in the `cf_values_br` field for each node.
		@param: node the current node
		'''
		pass
		# compute values using terminal_equity in terminal nodes
		# multiply by the pot
		# [[actions, players, ranges]]
		# strategy = [[actions x range]]
		# compute CFVs given the current strategy for this node
		# compute CFVs given the BR strategy for this node
		# counterfactual values weighted by the reach prob
		# compute CFV-BR values weighted by the reach prob


	def compute_values(self, root, starting_ranges):
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
		pass
		# 1.0 set the starting range
		# 2.0 check the starting ranges
		# 3.0 compute the values




#
