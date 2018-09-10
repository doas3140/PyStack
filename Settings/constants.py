'''
	Various constants used in DeepStack.
'''

from ..helper_classes import Players
from ..helper_classes import NodeTypes
from ..helper_classes import Actions
from ..helper_classes import ACPCActions

class Constants():
    def __init__(self):
        # the number of players in the game
        self.players_count = 2
        # the number of betting rounds in the game
        self.streets_count = 2

        # IDs for each player and chance
        self.players = Players()
		self.players.chance = -1
		self.players.P1 = 0
		self.players.P2 = 1

        # IDs for terminal nodes (either after a fold or call action) and
		# nodes that follow a check action
        self.node_types = NodeTypes()
		self.node_types.terminal_fold = -2 # terminal node following fold
		self.node_types.terminal_call = -1 # terminal node following call
		self.node_types.check = -1 # node for the chance player
		self.node_types.chance_node = 0 # node following check
		self.node_types.inner_node = 1 # any other node

        # IDs for fold and check/call actions
        self.actions = Actions()
		self.actions.fold = -2 #
		self.actions.ccall = -1 # (check/call)

        # String representations for actions in the ACPC protocol
        self.acpc_actions = ACPCActions()
		self.acpc_actions.fold = "fold"
		self.acpc_actions.ccall = "ccall" # (check/call)
		self.acpc_actions.raise = "raise"


constants = Constants()
