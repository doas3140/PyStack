'''
	Fills a game's public tree with a uniform strategy. In particular, fills
	the chance nodes with the probability of each outcome.

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
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..Game.card_tools import card_tools

class StrategyFilling():
	def __init__(self):
		pass


	def _fill_chance(self, node):
		''' Fills a chance node with the probability of each outcome.
		@param: node the chance node
		'''
	    pass
	    # we will fill strategy with an uniform probability, but it has to be
		# zero for hands that are not possible on corresponding board
		# setting probability of impossible hands to 0


	def _fill_uniformly(self, node):
		''' Fills a player node with a uniform strategy.
		@param: node the player node
		'''
		pass


	def _fill_uniform_dfs(self, node):
		'''Fills a node with a uniform strategy and recurses on the children.
		@param: node the node
		'''
	    pass


	def fill_uniform(self, tree):
		'''Fills a public tree with a uniform strategy.
		@param: tree a public tree for Leduc Hold'em or variant
		'''
		pass




strategy_filling = StrategyFilling()
