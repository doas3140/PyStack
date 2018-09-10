'''
	Builds a public tree for Leduc Hold'em or variants.
	Each node of the tree contains the following fields:
		* `node_type`: () int. an element of @{constants.node_types} (if applicable)
		* `street`: () int. the current betting round
		* `board`: (current_board_cards,) a possibly empty vector of board cards
		* `board_string`: str. a string representation of the board cards
		* `current_player`: () int. the player acting at the node
		* `bets`: (num_players,) the number of chips that each player has committed to the pot
		* `pot`: bets.min() half the pot size, equal to the smaller number in `bets`
		* `children`: [] a list of children nodes
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Game.card_tools import card_tools
from ..Game.card_to_string_conversion import card_to_string
from ..Tree.strategy_filling import StrategyFilling
from ..Game.bet_sizing import BetSizing

from ..helper_classes import Node

class PokerTreeBuilder():
	def __init__(self):
		pass


	def _get_children_nodes_transition_call(self, parent_node):
		''' Creates the child node after a call which transitions between
			betting rounds.
		@param: parent_node the node at which the transition call happens
		@return a list containing the child node
		'''
		pass


	def _get_children_nodes_chance_node(self, parent_node):
		''' Creates the children nodes after a chance node.
		@param: parent_node the chance node
		@return a list of children nodes
		'''
		pass


	def _fill_additional_attributes(self, node):
		''' Fills in additional convenience attributes which only depend
			on existing node attributes.
		@param: node
		'''
		pass


	def _get_children_player_node(self, parent_node):
		''' Creates the children nodes after a player node.
		@param: parent_node the chance node
		@return a list of children nodes
		'''
		pass
	    # 1.0 fold action
	    # 2.0 check action
	    # 2.0 terminal call - either last street or allin
	    # 3.0 bet actions

	def _get_children_nodes(self, parent_node):
		''' Creates the children after a node.
		@param: parent_node the node to create children for
		@return a list of children nodes
		'''
		pass
	    # transition call -> create a chance node
	    # inner nodes -> handle bet sizes


	def _build_tree_dfs(self, current_node):
		''' Recursively build the (sub)tree rooted at the current node.
		@param: current_node the root to build the (sub)tree from
		@return `current_node` after the (sub)tree has been built
		'''
		pass


	def build_tree(self, params):
		''' Builds the tree.
		@param: params table of tree parameters
		@return the root node of the built tree
		'''
		pass
	    # copy necessary stuff from the root_node not to touch the input




tree_builder = PokerTreeBuilder()
