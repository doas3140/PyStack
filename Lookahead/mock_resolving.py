'''
	Implements the re-solving interface used by @{resolving}
	with functions that do nothing.
	Used for debugging.
'''

from ..Lookahead.lookahead import Lookahead
from ..Lookahead.cfrd_gadget import CFRDGadget
# from ..Tree.tree_builder import PokerTreeBuilder # dont need here?
# from ..Tree.tree_visualiser import TreeVisualiser # dont need here?
from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..tools import tools
from ..Game.card_tools import card_tools
from ..Settings.game_settings import game_settings

class MockResolving():
	def __init__(self):
		pass

	def resolve_first_node(self, node, player_range, opponent_range):
		''' Does nothing
		@param: node the node to "re-solve"
		@param: [opt] player_range not used
		@param: [opt] opponent_range not used
		@see resolving.resolve_first_node
		'''
		pass


	def resolve(self, node, player_range, opponent_cfvs):
		''' Does nothing.
		@param: node the node to "re-solve"
		@param: [opt] player_range not used
		@param: [opt] opponent_cfvs not used
		@see resolving.resolve
		'''
		pass


	def get_possible_actions(self):
		''' Gives the possible actions at the re-solve node.
		@return the actions that can be taken at the re-solve node
		@see resolving.get_possible_actions
		'''
		pass


	def get_root_cfv(self):
		''' Returns an arbitrary vector.
		@return a vector of 1s
		@see resolving.get_root_cfv
		'''
		pass


	def get_action_cfv(self, action):
		''' Returns an arbitrary vector.
		@param: [opt] action not used
		@return a vector of 1s
		@see resolving.get_action_cfv
		'''
		pass


	def get_chance_action_cfv(self, player_action, board):
		''' Returns an arbitrary vector.
		@param: [opt] player_action not used
		@param: [opt] board not used
		@return a vector of 1s
		@see resolving.get_chance_action_cfv
		'''
		pass


	def get_action_strategy(self, action):
		''' Returns an arbitrary vector.
		@param: [opt] action not used
		@return a vector of 1s
		@see resolving.get_action_strategy
		'''
		pass




#
