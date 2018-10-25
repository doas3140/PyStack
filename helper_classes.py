'''
	Helper Classes
'''

class Players():
	def __init__(self):
		self.chance = 0
		self.P1 = 0
		self.P2 = 0

class NodeTypes():
	def __init__(self):
		self.terminal_fold = 0
		self.terminal_call = 0
		self.check = 0
		self.chance_node = 0
		self.inner_node = 0

class Actions():
	def __init__(self):
		self.fold = 0
		self.ccall = 0

class ACPCActions():
	def __init__(self):
		self.fold = ""
		self.ccall = ""
		self.raise_ = ""

class Node():
	def __init__(self):
		self.type = None # int
		self.node_type = None # int
		self.street = None # int
		self.board = None # np.array (current_board_cards,)
		self.board_string = None # str
		self.current_player = None # int
		self.bets = None # np.array (num_players,)
		self.pot = None # int
		self.children = [] # list
		self.terminal = None # boolean
		self.parent = None # Node
		self.actions = None # np.array (len(children),)
		self.strategy = None # np.array (len(children), CC)
		# cfr
		self.iter_weight_sum = None # np.array (CC,)
		self.regrets = None # np.array (AC,CC)
		self.possitive_regrets = None # np.array (AC,CC)
		self.cf_values = None # np.array (PC,CC)
		self.ranges_absolute = None # np.array (PC,CC)


	# def copy(self):
	# 	new_node = Node()
	# 	if self.node_type is not None:
	# 		new_node.node_type = 		self.node_type
	# 	if self.street is not None:
	# 		new_node.street = 			self.street
	# 	if self.board is not None:
	# 		new_node.board = 			self.board.copy()
	# 	if self.board_string is not None:
	# 		new_node.board_string = 	self.board_string
	# 	if self.current_player is not None:
	# 		new_node.current_player = 	self.current_player
	# 	if self.bets is not None:
	# 		new_node.bets = 			self.bets.copy()
	# 	if self.pot is not None:
	# 		new_node.pot = 				self.pot
	# 	if self.children is not None:
	# 		new_node.children = 		self.children
	# 	if self.terminal is not None:
	# 		new_node.terminal = 		self.terminal
	# 	return new_node

class TreeParams():
	def __init__(self):
		self.root_node = None # Node obj
		self.limit_to_street = None # boolean
		self.bet_sizing = None

	def init_root_node(self): # from Player.continual_resolving lua file
		import numpy as np
		from Settings.constants import constants
		from Settings.arguments import arguments
		first_node = Node()
		first_node.board = np.zeros([], dtype=int)
		first_node.street = 1
		first_node.current_player = constants.players.P1
		first_node.bets = np.array([arguments.ante, arguments.ante], dtype=int)
		self.root_node = first_node
		self.limit_to_street = True



#
