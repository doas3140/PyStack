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

class Node():
	def __init__(self):
		self.type = None # int
		self.node_type = None # int
		self.street = None # int (the current betting round)
		self.board = None # [0-5] (vector of cards on board)
		self.board_string = None # str
		self.current_player = None # int
		self.bets = None # [P] (current bets from both players)
		self.num_bets = None # int (used for additional betting in first round)
		self.pot = None # int ( pot = max(self.bets) )
		self.children = [] # [Node,...] (list of nodes)
		self.terminal = None # boolean (is this node terminal)
		self.parent = None # Node
		self.actions = None # [len(children)] (available bet sizes + call + fold actions)
		self.strategy = None # [len(children), I] (strategy for each hand)
		# cfr
		self.iter_weight_sum = None # [I]
		self.regrets = None # [A,I]
		self.possitive_regrets = None # [A,I] (clipped self.regrets)
		self.cf_values = None # [P,I]
		self.ranges = None # [P,I]

class TreeParams():
	def __init__(self):
		self.root_node = None # Node obj
		self.limit_to_street = None # boolean

class ResolvingParams():
	def __init__(self):
		self.node = None # Node
		self.range = None # [I] (probability vector over the player's private hands at the node)
		self.player = None # int
		self.cf_values = None # [I] (vector of opponent counterfactual values at the node)
		self.resolving = None # Resolving (object which was used to re-solve the last player node)
		self.our_last_action = None # (the action taken by the player at their last node)
		self.opponent_range = None # [I] (probability vector over the player's private hands at the node)

class LookaheadResults():
	def __init__(self):
		self.strategy = None				# [A{0}, b, I]
		self.achieved_cfvs = None			# [b, P, I]
		self.root_cfvs = None				# [b, I]
		self.root_cfvs_both_players = None  # [b, P, I]
		self.children_cfvs = None 			# [A{0}, b, I]
		# vars below are used to store next round cfvs
		self.next_street_cfvs = None		# [b x trans_nodes, B, P, I]
		self.next_boards = None				# [B, 0-5]
		self.actions = None					# [A] (bets)
		self.action_to_index = None			# {'bet size':'next_street_cfvs index'}
		self.next_round_pot_sizes = None	# [b x trans_nodes, B]

	# def __str__(self):
	# 	return 'strat\n {} \ncfvs\n {} \nroot_cfvs\n {} \nboth_P_root_cfvs\n {} \nchildren_cfvs\n {}'. \
	# 						format(  self.strategy, self.achieved_cfvs, \
	# 	 							 self.root_cfvs, self.root_cfvs_both_players, \
	# 								 self.children_cfvs )

class LookaheadLayer():
	def __init__(self):
		# _compute_tree_structures
		# scalars
		self.num_bets = None # [0 - d]
		self.num_nonallin_bets = None # [0 - d]
		self.num_terminal_actions = None # [0 - d]
		self.num_actions = None # [0 - d]
		# _compute_structure
		# scalars
		self.acting_player = None # [1 - d+1]
		self.num_nonterminal_nodes = None # [1 - d]
		self.num_nonterminal_nonallin_nodes = None # [1 - d]
		self.num_all_nodes = None # [1 - d]
		self.num_allin_nodes = None # [1 - d]
		self.next_street_boxes_inputs = None # [1 - d]
		self.next_street_boxes_outputs = None # [d - d]
		# construct_data_structures
		# [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
		self.ranges = None # [0 - d]
		self.pot_size = None # [0 - d]
		self.cfvs = None # [0 - d]
		self.cfvs_avg = None # [0 - d]
		# [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I]
		self.strategies_avg = None # [0 - d]
		self.current_strategy = None # [0 - d]
		self.regrets = None # [0 - d]
		self.empty_action_mask = None # [0 - d]
		# for terminal equity (2,)
		self.term_call_idx = None # [1 - d]
		self.term_fold_idx = None # [1 - d]
		# _construct_transition_boxes
		self.indices = None # (2,) [1 - d]



#
