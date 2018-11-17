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
		self.raise_ = 0

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
		self.num_bets = None
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
		# lookahead
		self.lookahead_coordinates = None # np.array [action_id, parent_id, grandparent_id]


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

class ResolvingParams():
	def __init__(self):
		self.node = None # node
		self.range = None # p2_range
		self.player = None # player
		self.cf_values = None # cf_values
		self.resolving = None
		self.our_last_action = None
		self.opponent_range = None

class LookaheadResults():
	def __init__(self):
		self.strategy = None
		self.achieved_cfvs = None
		self.root_cfvs = None
		self.root_cfvs_both_players = None
		self.children_cfvs = None

	def __str__(self):
		s = 'strat\n {} \ncfvs\n {} \nroot_cfvs\n {} \nboth_P_root_cfvs\n {} \nchildren_cfvs\n {}'. \
							format(  self.strategy, self.achieved_cfvs, \
		 							 self.root_cfvs, self.root_cfvs_both_players, \
									 self.children_cfvs )
		return s

class Lookahead():
	def __init__(self):
		self.ccall_action_index = None # int
		self.fold_action_index = None # int
		self.tree = None # Node obj
		self.next_street_boxes = None # dict [int->NextRoundValue obj]
		self.regret_epsilon = None # const int
		self.acting_player = None # arr[depth+1]
		self.depth = None # int
		self.bets_count = None # dict
		self.nonallinbets_count = None # dict
		self.terminal_actions_count = None # dict
		self.actions_count = None # dict
		# node counts
		self.nonterminal_nodes_count = None # dict
		self.nonterminal_nonallin_nodes_count = None # dict
		self.all_nodes_count = None # dict
		self.terminal_nodes_count = None # dict
		self.allin_nodes_count = None # dict
		self.inner_nodes_count = None # dict
		# lookahead main data structures
  		# all the structures are per-layer tensors, that is,
		# each layer holds the data in n-dimensional tensors
		self.pot_size = None # dict
		self.ranges_data = None # dict
		self.average_strategies_data = None # dict
		self.current_strategy_data = None # dict
		self.cfvs_data = None # dict
		self.average_cfvs_data = None # dict
		self.regrets_data = None # dict
		self.current_regrets_data = None # dict
		self.positive_regrets_data = None # dict
		self.placeholder_data = None # dict
		self.regrets_sum = None # dict
		self.empty_action_mask = None # dict
		# used to hold and swap inner (nonterminal) nodes when doing some transpose operations
		self.inner_nodes = None # dict
		self.inner_nodes_p1 = None # dict
		self.swap_data = None # dict




#
