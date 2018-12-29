'''
	Builds the internal data structures of a @{lookahead|Lookahead} object.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from NeuralNetwork.next_round_value import NextRoundValue, get_next_round_value
from helper_classes import LookaheadLayer


class LookaheadBuilder():
	def __init__(self, lookahead):
		self.lookahead = lookahead


	def _construct_transition_boxes(self):
		''' Builds the neural net query boxes which estimate
			counterfactual values at depth-limited states of the lookahead.
		'''
		if self.lookahead.tree.street == constants.streets_count:
			return

		self.lookahead.num_pot_sizes = 0
		# create the optimized data structures for batching next_round_value
		for d in range(1,self.lookahead.depth):
			# layers[d].pot_size - [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			num_grandparent_bets = self.lookahead.layers[d].pot_size[1].shape[0]
			if d == 1 and self.lookahead.first_call_transition:
				before = self.lookahead.num_pot_sizes
				self.lookahead.num_pot_sizes += 1
				self.lookahead.layers[d].indices = np.array([before, self.lookahead.num_pot_sizes], dtype=arguments.int_dtype)
			elif num_grandparent_bets > 1:
				before = self.lookahead.num_pot_sizes
				num_nonterminal_nonallin_grandparents = self.lookahead.layers[d].pot_size[1].shape[1]
				num_grandparent_nonallin_bets = self.lookahead.layers[d].pot_size[1].shape[0] - 1
				num_nonterminal_nonallin_parents = num_nonterminal_nonallin_grandparents * num_grandparent_nonallin_bets
				self.lookahead.num_pot_sizes += num_nonterminal_nonallin_parents
				self.lookahead.layers[d].indices = np.array([before, self.lookahead.num_pot_sizes], dtype=arguments.int_dtype)

		if self.lookahead.num_pot_sizes == 0:
			return

		self.lookahead.next_round_pot_sizes = np.zeros([self.lookahead.num_pot_sizes], dtype=arguments.dtype)
		self.lookahead.action_to_index = {}

		for d in range(1,self.lookahead.depth):
			if self.lookahead.layers[d].indices is not None:
				p_start, p_end = (0,1) if d == 1 else (0,-1) # parent indices
				pot_sizes = self.lookahead.layers[d].pot_size[ 1, p_start:p_end, : , 0, 0, 0 ].reshape([-1])
				self.lookahead.next_round_pot_sizes[ self.lookahead.layers[d].indices[0]:self.lookahead.layers[d].indices[1] ] = pot_sizes.copy()
				if d == 1:
					assert(self.lookahead.layers[1].indices[0] == self.lookahead.layers[1].indices[1]-1)
					self.lookahead.action_to_index[constants.actions.ccall] = self.lookahead.layers[1].indices[0]
				elif d == 2:
					assert(self.lookahead.layers[2].pot_size[1, p_start:p_end].shape[1] == 1) # bad num_indices
					num_root_node_bets = self.lookahead.layers[2].pot_size.shape[1]
					for action_idx in range(num_root_node_bets):
						action = self.lookahead.parent_action_id[action_idx]
						assert(action not in self.lookahead.action_to_index)
						self.lookahead.action_to_index[action] = self.lookahead.layers[d].indices[0] + action_idx

		street, board = self.lookahead.tree.street, self.lookahead.terminal_equity.board
		self.lookahead.cfvs_approximator = get_next_round_value(street) # (loads preloaded models)
		# init input/output variables in NextRoundValue
		self.lookahead.cfvs_approximator.init_computation(board, self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)


	def _compute_structure(self):
		''' Computes the number of nodes at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
		'''
		assert(self.lookahead.tree.street >= 1 and self.lookahead.tree.street <= constants.streets_count)
		layers = self.lookahead.layers
		# which player acts at particular depth
		layers[0].acting_player = 0
		for d in range(1, self.lookahead.depth+1):
			layers[d].acting_player = 1 - layers[d-1].acting_player
		# compute the node counts
		layers[0].num_nonterminal_nodes = 1
		layers[1].num_nonterminal_nodes = layers[0].num_bets
		# self.lookahead.num_nonterminal_nonallin_nodes[0] = 1
		layers[0].num_nonterminal_nonallin_nodes = 1
		layers[1].num_nonterminal_nonallin_nodes = layers[1].num_nonterminal_nodes - 1
		layers[0].num_all_nodes = 1
		layers[1].num_all_nodes = layers[0].num_actions
		layers[0].num_allin_nodes = 0
		layers[1].num_allin_nodes = 1
		for d in range(1, self.lookahead.depth):
			layers[d+1].num_all_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_bets * layers[d].num_actions
			layers[d+1].num_allin_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_bets * 1
			layers[d+1].num_nonterminal_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_nonallin_bets * layers[d].num_bets
			layers[d+1].num_nonterminal_nonallin_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_nonallin_bets * layers[d].num_nonallin_bets


	def construct_data_structures(self):
		''' Builds the tensors that store lookahead data during re-solving '''
		PC, HC, batch_size = constants.players_count, constants.hand_count, self.lookahead.batch_size
		layers = self.lookahead.layers
		# lookahead main data structures
		# all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
		# create the data structure for the first two layers
		# data structures [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
		# [actions, parent_action, grandparents, batch, players, range]
		layers[0].ranges = np.full([1, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		layers[1].ranges = np.full([layers[0].num_actions, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		layers[0].pot_size = np.zeros_like(layers[0].ranges)
		layers[1].pot_size = np.zeros_like(layers[1].ranges)
		layers[0].cfvs = np.zeros_like(layers[0].ranges)
		layers[1].cfvs = np.zeros_like(layers[1].ranges)
		layers[0].cfvs_avg = np.zeros_like(layers[0].ranges)
		layers[1].cfvs_avg = np.zeros_like(layers[1].ranges)
		# data structures for one player [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I]
		layers[0].strategies_avg = None
		layers[1].strategies_avg = np.zeros([layers[0].num_actions, 1, 1, batch_size, HC], dtype=arguments.dtype)
		layers[0].current_strategy = None
		layers[1].current_strategy = np.zeros_like(layers[1].strategies_avg)
		layers[0].regrets = None
		layers[1].regrets = np.zeros_like(layers[1].strategies_avg)
		layers[0].empty_action_mask = None
		layers[1].empty_action_mask = np.ones_like(layers[1].strategies_avg)
		# create the data structures for the rest of the layers
		for d in range(2, self.lookahead.depth):
			# data structures [A{d-1}, B{d-2}, NTNAN{d-2}, b, P, I]
			layers[d].ranges = np.zeros([layers[d-1].num_actions, layers[d-2].num_bets, layers[d-2].num_nonterminal_nonallin_nodes, batch_size, PC, HC], dtype=arguments.dtype)
			layers[d].cfvs = layers[d].ranges.copy()
			layers[d].pot_size = np.full_like(layers[d].ranges, arguments.stack)
			# data structures [A{d-1}, B{d-2}, NTNAN{d-2}, b, 1, I]
			layers[d].strategies_avg = np.zeros([layers[d-1].num_actions, layers[d-2].num_bets, layers[d-2].num_nonterminal_nonallin_nodes, batch_size, HC], dtype=arguments.dtype)
			layers[d].current_strategy = layers[d].strategies_avg.copy()
			layers[d].regrets = np.full_like(layers[d].strategies_avg, constants.regret_epsilon)
			layers[d].empty_action_mask = np.ones_like(layers[d].strategies_avg)
		# save indexes of nodes that are terminal, so we can use them to calculate rewards from terminal equity in one batch
		self.lookahead.num_term_call_nodes = 0
		self.lookahead.num_term_fold_nodes = 0
		# calculate term_call_indices
		for d in range(1,self.lookahead.depth):
			assert(layers[d].ranges[1][-1].shape[0] == layers[d].ranges[1].shape[1])
			if d > 1 or self.lookahead.first_call_terminal:
				if self.lookahead.tree.street != constants.streets_count:
					before = self.lookahead.num_term_call_nodes
					num_nonterminal_nonallin_grandparents = layers[d].ranges[1][-1].shape[0]
					self.lookahead.num_term_call_nodes += num_nonterminal_nonallin_grandparents
					layers[d].term_call_idx = np.array([before, self.lookahead.num_term_call_nodes], dtype=arguments.int_dtype)
				else:
					before = self.lookahead.num_term_call_nodes
					num_grandparent_bets = layers[d].ranges[1].shape[0]
					num_nonterminal_nonallin_grandparents = layers[d].ranges[1].shape[1]
					num_nonterminal_parents = num_nonterminal_nonallin_grandparents * num_grandparent_bets
					self.lookahead.num_term_call_nodes += num_nonterminal_parents
					layers[d].term_call_idx = np.array([before, self.lookahead.num_term_call_nodes], dtype=arguments.int_dtype)
		# calculate term_fold_indices
		for d in range(1,self.lookahead.depth):
			before = self.lookahead.num_term_fold_nodes
			num_grandparent_bets = layers[d].ranges[0].shape[0]
			num_nonterminal_nonallin_grandparents = layers[d].ranges[0].shape[1]
			num_nonterminal_parents = num_nonterminal_nonallin_grandparents * num_grandparent_bets
			self.lookahead.num_term_fold_nodes += num_nonterminal_parents
			layers[d].term_fold_idx = np.array([before, self.lookahead.num_term_fold_nodes], dtype=arguments.int_dtype)


	def set_datastructures_from_tree_dfs(self, node, depth, action_id, parent_id, gp_id, cur_action_id, parent_action_id=None):
		''' Traverses the tree to fill in lookahead data structures that
			summarize data contained in the tree.
			(saves pot sizes and numbers of actions at each lookahead state)
		@param: Node :node the current node of the public tree
		@param: int  :depth of the current node
		@param: int  :index of the action that led to this node
		@param: int  :index of the current node's parent
		@param: int  :index of the current node's grandparent
		@param: int  :parent's action
		@param: int  :grandparent's action
		'''
		# fill the potsize
		assert(node.pot)
		self.lookahead.layers[depth].pot_size[ action_id, parent_id, gp_id, : , : ] = node.pot
		if depth == 2 and cur_action_id == constants.actions.ccall:
			self.lookahead.parent_action_id[parent_id] = parent_action_id
		# node.lookahead_coordinates = np.array([action_id, parent_id, gp_id], dtype=arguments.dtype)
		# transition call cannot be allin call
		if node.current_player == constants.players.chance:
			num_nonallin_bets = self.lookahead.layers[depth-2].num_nonallin_bets if depth > 1 else 1
			assert(parent_id <= num_nonallin_bets)
		if depth < self.lookahead.depth + 1:
			gp_num_nonallin_bets = self.lookahead.layers[depth-2].num_nonallin_bets if depth > 1 else 1
			p_num_terminal_actions = self.lookahead.layers[depth-1].num_terminal_actions if depth > 0 else 0
			# compute next coordinates for parent and grandparent
			next_parent_id = action_id - p_num_terminal_actions
			next_gp_id = gp_id * gp_num_nonallin_bets + parent_id
			if (not node.terminal) and (node.current_player != constants.players.chance):
				# parent is not an allin raise
				if depth > 1:
					assert(parent_id <= self.lookahead.layers[depth-2].num_nonallin_bets)
				# do we need to mask some actions for that node? (that is, does the node have fewer children than the max number of children for any node on this layer)
				if len(node.children) < self.lookahead.layers[depth].num_actions:
					# we need to mask nonexisting padded bets
					assert(depth > 0)
					num_terminal_actions = self.lookahead.layers[depth].num_terminal_actions
					assert(num_terminal_actions == 2)
					existing_num_bets = len(node.children) - num_terminal_actions
					# allin situations
					if existing_num_bets == 0:
						if depth > 0:
							assert(action_id == self.lookahead.layers[depth-1].num_actions-1)
					for action_id in range(num_terminal_actions): # go deeper
						next_node, next_action = node.children[action_id], node.actions[action_id]
						self.set_datastructures_from_tree_dfs(next_node, depth+1, action_id, next_parent_id, next_gp_id, next_action, cur_action_id)
					# we need to make sure that even though there are fewer actions,
					# the last action/allin is has the same last index as if we had full number of actions
					existing_num_nonallin_bets = existing_num_bets - 1
					for b in range(2, existing_num_nonallin_bets): # 2, because 0 - Fold, 1 - Call, 2 and more - bets, N - all-in
						action_id = len(node.children) - b
						next_node, next_action = node.children[action_id], node.actions[action_id]
						action_id = self.lookahead.layers[depth].num_actions - b # we manually set the action_id as the last action (allin)
						self.set_datastructures_from_tree_dfs(next_node, depth+1, action_id, next_parent_id, next_gp_id, next_action, cur_action_id)
					# mask out empty actions
					# self.lookahead.layers[depth+1].empty_action_mask.shape[0] == self.lookahead.layers[depth].num_actions
					a = self.lookahead.layers[depth+1].empty_action_mask.shape[0] - existing_num_bets # A{d} - node.A + TA{d}
					self.lookahead.layers[depth+1].empty_action_mask[ num_terminal_actions:a, next_parent_id, next_gp_id, : ] = 0
				else:
					# node has full action count, easy to handle
					for action_id in range(len(node.children)): # go deeper
						next_node, next_action = node.children[action_id], node.actions[action_id]
						self.set_datastructures_from_tree_dfs(next_node, depth+1, action_id, next_parent_id, next_gp_id, next_action, cur_action_id)


	def build_from_tree(self, tree):
		''' Builds the lookahead's internal data structures using the public tree
		@param: Node :public tree used to construct the lookahead
		'''
		self.lookahead.tree = tree
		self.lookahead.depth = tree.depth
		# init layers
		self.lookahead.layers = []
		for d in range(0,self.lookahead.depth+1):
			self.lookahead.layers.append( LookaheadLayer() )
		# per layer information about tree actions
		# per layer actions are the max number of actions for any of the nodes on the layer
		self.lookahead.first_call_terminal = self.lookahead.tree.children[1].terminal
		self.lookahead.first_call_transition = self.lookahead.tree.children[1].current_player == constants.players.chance
		self.lookahead.first_call_check = (not self.lookahead.first_call_terminal) and (not self.lookahead.first_call_transition)
		self._compute_tree_structures([tree], current_depth=0)
		# construct the initial data structures using the bet counts
		self._compute_structure()
		self.construct_data_structures()
		# action ids for first
		self.lookahead.parent_action_id = {}
		# traverse the tree and fill the datastructures (pot sizes and non-existin actions masks)
		self.set_datastructures_from_tree_dfs(tree, depth=0, action_id=0, parent_id=0, gp_id=0, cur_action_id=-100)
		# we mask out fold as a possible action when check is for free, due to: fewer actions means faster convergence
		if self.lookahead.tree.bets[0] == self.lookahead.tree.bets[1]:
			self.lookahead.layers[1].empty_action_mask[0].fill(0)
		# construct the neural net query boxes
		self._construct_transition_boxes()


	def _compute_tree_structures(self, current_layer, current_depth):
		''' Computes the maximum number of actions at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
			The maximum number of actions is used so that every node at that
			depth can fit in the same tensor.
		@param: [Node,...] :list of tree nodes at the current depth
		@param: int        :depth of the current tree nodes
		'''
		layer_num_actions, layer_num_terminal_actions = 0, 0
		next_layer = []
		for n in range(len(current_layer)):
			node = current_layer[n]
			layer_num_actions = max(layer_num_actions, len(node.children))
			node_num_terminal_actions = 0
			for c in range(len(current_layer[n].children)):
				if node.children[c].terminal or node.children[c].current_player == constants.players.chance:
					node_num_terminal_actions += 1
			layer_num_terminal_actions = max(layer_num_terminal_actions, node_num_terminal_actions)
			# add children of the node to the next layer for later pass of BFS
			if not node.terminal:
				for c in range(len(node.children)):
					next_layer.append(node.children[c])
		assert((layer_num_actions == 0) == (len(next_layer) == 0))
		assert((layer_num_actions == 0) == (current_depth == self.lookahead.depth - 1))
		# set action and bet counts
		self.lookahead.layers[current_depth].num_bets = layer_num_actions - layer_num_terminal_actions
		self.lookahead.layers[current_depth].num_nonallin_bets = self.lookahead.layers[current_depth].num_bets - 1 # remove allin
		# if no allin...
		if layer_num_actions == 2:
			assert(layer_num_actions == layer_num_terminal_actions)
			self.lookahead.layers[current_depth].num_nonallin_bets = 0
		self.lookahead.layers[current_depth].num_terminal_actions = layer_num_terminal_actions
		self.lookahead.layers[current_depth].num_actions = layer_num_actions
		if len(next_layer) > 0:
			assert(layer_num_actions >= 2)
			# go deeper
			self._compute_tree_structures(next_layer, current_depth + 1)




#
