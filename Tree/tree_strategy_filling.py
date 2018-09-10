'''
	Recursively performs continual re-solving at every node of a public tree to
	generate the DeepStack strategy for the entire game.

	A strategy is represented at each public node by a (N,K) tensor where:
	* N is the number of possible child nodes.
	* K is the number of information sets for the active player in the public
	node. For the Leduc Hold'em variants we implement, there is one for each
	private card that the player could hold.

	For a player node, `strategy[i][j]` gives the probability of taking the
	action that leads to the `i`th child when the player holds the `j`th card.

	For a chance node, `strategy[i][j]` gives the probability of reaching the
	`i`th child for either player when that player holds the `j`th card.
'''

from ..Settings.arguments import arguments
from ..Game.card_tools import card_tools
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
# from ..Lookahead.mock_resolving import MockResolving # dont need here?
from ..Lookahead.resolving import Resolving

class TreeStrategyFilling():
	def __init__(self):
		self.board_count = card_tools.get_boards_count()


	def _fill_chance(self, node):
		''' Fills all chance nodes of a subtree with the probability of each outcome.
		@param: node the root of the subtree
		'''
		pass
		# chance node, we will fill uniform strategy
		# works only for chance node at start of second round
		# we will fill strategy with an uniform probability,
		# but it has to be zero for hands that are not possible on
		# corresponding board
		# setting strategy for impossible hands to 0


	def _fill_uniformly(self, node, player):
		''' Recursively fills a subtree with a uniform random strategy
			for the given player.
			Used in sections of the game to which the player doesn't play.
		@param: node the root of the subtree
		@param: player the player which is given the uniform random strategy
		'''
		pass
		# fill uniform strategy


	def _process_opponent_node(self, params):
		''' Recursively fills a player's strategy for the subtree rooted at an
			opponent node.
		@param params tree walk parameters (see @{_fill_strategies_dfs})
		'''
		pass
		# node, player, range, cf_values, strategy_computation, our_last_action
		# when opponent plays, we will do nothing except sending cf_values to the child nodes


	def _fill_starting_node(self, node, player, p1_range, p2_range):
		''' Recursively fills a player's strategy in a tree.
		@param: node the root of the tree
		@param: player the player to calculate a strategy for
		@param: p1_range a probability vector of the first player's
				private hand at the root
		@param: p2_range a probability vector of the second player's
				private hand at the root
		'''
		pass
		# re-solving the node
		# check which player plays first
		# opponent plays in this node. we need only cf-values at the beginning and we will just copy them


	def _fill_player_node(self, params):
		''' Recursively fills a player's strategy for the subtree rooted at a
			player node.
			Re-solves to generate a strategy for the player node.
		@param: params tree walk parameters (see @{_fill_strategies_dfs})
		'''
		pass
		# now player plays, we have to compute his strategy
		# we will send opponent range to adjust range also in our
		# second action in the street


	def _fill_computed_node(self, node, player, range, resolving):
		''' Recursively fills a player's strategy for the subtree
			rooted at a player node.
		@param: node the player node
		@param: player the player to fill the strategy for
		@param: range a probability vector giving the player's range at the node
		@param: resolving a @{resolving|Resolving} object which has been used to
				re-solve the node
		'''
		pass
		# find which bets are used by player
		# there has to be exactly one equivalent bet
		# check if terminal actions are used and if all player bets are used
		# fill the strategy
		# we need to compute all values and ranges before dfs call, becasue
    	# re-solving will be built from different node in the recursion
		# in first cycle, fill nodes we do not play in and fill strategies and cf-values
		# check if the bet is possible
		# compute ranges for each action
		# normalize the ranges
		# in second cycle, run dfs computation


	def _process_chance_node(self, params):
		''' Recursively fills a player's strategy for the subtree rooted at a
			chance node.
		@param: params tree walk parameters (see @{_fill_strategies_dfs})
		'''
		pass
		# on chance node we need to recompute values in next round
		# computing cf_values for the child node
		# we need to remove impossible hands from the range and then renormalize it
		# weight should be single number
		# we should never touch same re-solving again after the chance action,
		# set it to nil


	def _fill_strategies_dfs(self, params):
		''' Recursively fills a player's strategy for a subtree.
		@param: params a table of tree walk parameters with the following fields:
				* `node`: the root of the subtree
				* `player`: the player to fill the strategy for
				* `range`: a probability vector over the player's private hands
						   at the node
				* `cf_values`: a vector of opponent counterfactual values
							   at the node
				* `resolving`: a @{resolving|Resolving} object which was used to
							   re-solve the last player node
				* `our_last_action`: the action taken by the player
									 at their last node
		'''
		pass


	def fill_strategies(self, root, player, p1_range, p2_range):
		''' Fills a tree with a player's strategy generated with continual re-solving.
			Recursively does continual re-solving on every node of the tree
			to generate the strategy for that node.
		@param: root the root of the tree
		@param: player the player to fill the strategy for
		@param: p1_range a probability vector over the first player's
				private hands at the root of the tree
		@param: p2_range a probability vector over the second player's
				private hands at the root of the tree
		'''
		pass


	def fill_uniform_strategy(self, root):
		''' Fills a tree with uniform random strategies for both players.
		@param: root the root of the tree
		'''
		pass




#
