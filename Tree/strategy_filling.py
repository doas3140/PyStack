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
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from Game.card_tools import card_tools

class StrategyFilling():
	def __init__(self):
		pass


	def _fill_chance(self, node):
		''' Fills a chance node with the probability of each outcome.
		@param: node the chance node
		'''
		CC = game_settings.card_count
		assert (not node.terminal)
		# filling strategy
		# we will fill strategy with an uniform probability, but it has to be
		# zero for hands that are not possible on corresponding board
		node.strategy = np.zeros([len(node.children), CC])
		# setting probability of impossible hands to 0
		for i in range(len(node.children)):
			child_node = node.children[i]
			mask = card_tools.get_possible_hand_indexes(child_node.board)
			node.strategy[i].fill(0)
			# remove 2 because each player holds one card
			node.strategy[i][mask] = 1.0 / (CC - 2)


	def _fill_uniformly(self, node):
		''' Fills a player node with a uniform strategy.
		@param: node the player node
		'''
		CC = game_settings.card_count
		assert (node.current_player == constants.players.P1\
				or node.current_player == constants.players.P2)
		if node.terminal:
			return
		node.strategy = np.zeros([len(node.children), CC])
		node.strategy.fill(1.0 / len(node.children))


	def _fill_uniform_dfs(self, node):
		'''Fills a node with a uniform strategy and recurses on the children.
		@param: node the node
		'''
		if node.current_player == constants.players.chance:
			self._fill_chance(node)
		else:
			self._fill_uniformly(node)
		for i in range(len(node.children)):
			self._fill_uniform_dfs(node.children[i])


	def fill_uniform(self, tree):
		'''Fills a public tree with a uniform strategy.
		@param: tree a public tree for Leduc Hold'em or variant
		'''
		self._fill_uniform_dfs(tree)



strategy_filling = StrategyFilling()
