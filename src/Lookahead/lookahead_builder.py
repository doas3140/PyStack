'''
	Builds the internal data structures of a @{lookahead|Lookahead} object.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from NeuralNetwork.next_round_value import NextRoundValue
from NeuralNetwork.value_nn import ValueNn
from helper_classes import LookaheadLayer

# NEURAL_NET = {}
# AUX_NET = None
# NEXT_ROUND_PRE = None

class LookaheadBuilder():
	def __init__(self, lookahead):
		self.lookahead = lookahead
		self.lookahead.ccall_action_index = 1
		self.lookahead.fold_action_index = 2

	def _construct_transition_boxes(self):
		''' Builds the neural net query boxes which estimate
			counterfactual values at depth-limited states of the lookahead.
		'''
		if self.lookahead.tree.street == constants.streets_count:
			return
		# # load neural net (of next layer) if not already loaded
		if self.lookahead.tree.street == 1:
			nn = ValueNn(self.lookahead.tree.street, pretrained_weights=True, verbose=0)
		else:
			next_street = self.lookahead.tree.street + 1
			nn = ValueNn(next_street, pretrained_weights=True, verbose=0)

		if self.lookahead.tree.street == 1:
			self.lookahead.next_street_boxes = NextRoundValuePre(nn, aux_net, self.lookahead.terminal_equity.board)
		else:
			self.lookahead.next_street_boxes = NextRoundValue(nn, self.lookahead.terminal_equity.board)

		# if self.lookahead.tree.street in NEURAL_NET:
		# 	nn = NEURAL_NET[self.lookahead.tree.street]
		# else:
		# 	nn = ValueNn(pretrained_weights=True, verbose=0)
		# NEURAL_NET[self.lookahead.tree.street] = nn

		# if self.lookahead.tree.street == 1:
		# 	if AUX_NET is None:
		# 		aux_net = ValueNn(self.lookahead.tree.street, True)
		# 		AUX_NET = aux_net
		# 	else:
		# 		aux_net = AUX_NET

		# if self.lookahead.tree.street == 1:
		# 	if NEXT_ROUND_PRE is None:
		# 		self.lookahead.next_street_boxes = NEXT_ROUND_PRE
		# 	else:
		# 		self.lookahead.next_street_boxes = NextRoundValuePre(nn, aux_net, self.lookahead.terminal_equity.board)
		# 	NEXT_ROUND_PRE = self.lookahead.next_street_boxes
		# else:
		# 	self.lookahead.next_street_boxes = NextRoundValue(nn, self.lookahead.terminal_equity.board)

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
				if d <= 2:
					if d == 1:
						assert(self.lookahead.layers[d].indices[0] == self.lookahead.layers[d].indices[1])
						self.lookahead.action_to_index[constants.actions.ccall] = self.lookahead.layers[d].indices[0]
					else:
						assert(self.lookahead.layers[d].pot_size[1, p_start:p_end].shape[1] == 1) # bad num_indices
						for parent_action_idx in range(0, self.lookahead.layers[d].pot_size[1].shape[0]):
							action_id = self.lookahead.parent_action_id[parent_action_idx]
							assert(action_id not in self.lookahead.action_to_index)
							self.lookahead.action_to_index[action_id] = self.lookahead.layers[d].indices[0] + parent_action_idx

		# print(self.lookahead.next_round_pot_sizes)
		# print(self.lookahead.num_pot_sizes)
		# # for d in range(1,self.lookahead.depth):
		# # 	print(self.lookahead.layers[d].indices)
		# # for d in range(1,self.lookahead.depth):
		# # 	print(self.lookahead.layers[d].pot_size)
		print(self.lookahead.action_to_index)



		if constants.actions.ccall not in self.lookahead.action_to_index:
			print(self.lookahead.action_to_index)
			print(self.lookahead.parent_action_id)
			assert(False)

		PC, HC = constants.players_count, constants.hand_count
		self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)
		# self.lookahead.next_street_boxes_inputs = np.zeros([self.lookahead.num_pot_sizes, self.lookahead.batch_size, PC, HC], dtype=arguments.dtype)
		# self.lookahead.next_street_boxes_outputs = self.lookahead.next_street_boxes_inputs.copy()


	def _compute_structure(self):
		''' Computes the number of nodes at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
		'''
		assert(self.lookahead.tree.street >= 1 and self.lookahead.tree.street <= constants.streets_count)
		self.lookahead.regret_epsilon = 1.0 / 1000000000
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
		# neural network input and output boxes
		self.lookahead.next_street_boxes_inputs = {}
		self.lookahead.next_street_boxes_outputs = {}
		for d in range(1, self.lookahead.depth):
			layers[d+1].num_all_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_bets * layers[d].num_actions
			layers[d+1].num_allin_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_bets * 1
			layers[d+1].num_nonterminal_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_nonallin_bets * layers[d].num_bets
			layers[d+1].num_nonterminal_nonallin_nodes = layers[d-1].num_nonterminal_nonallin_nodes * layers[d-1].num_nonallin_bets * layers[d].num_nonallin_bets
			layers[d].next_street_boxes_inputs = None
			layers[d].next_street_boxes_outputs = None


	def construct_data_structures(self):
		''' Builds the tensors that store lookahead data during re-solving.
		'''
		PC, HC, batch_size = constants.players_count, constants.hand_count, self.lookahead.batch_size
		layers = self.lookahead.layers
		# lookahead main data structures
		# all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
		# create the data structure for the first two layers
		# data structures [actions x parent_action x grandparent_id x batch x players x range]
		layers[0].ranges = np.full([1, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		layers[1].ranges = np.full([layers[0].num_actions, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		layers[0].pot_size = np.zeros_like(layers[0].ranges)
		layers[1].pot_size = np.zeros_like(layers[1].ranges)
		layers[0].cfvs = np.zeros_like(layers[0].ranges)
		layers[1].cfvs = np.zeros_like(layers[1].ranges)
		layers[0].cfvs_avg = np.zeros_like(layers[0].ranges)
		layers[1].cfvs_avg = np.zeros_like(layers[1].ranges)
		# data structures for one player [actions x parent_action x grandparent_id x 1 x range]
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
			# data structures [actions x parent_action x grandparent_id x batch x players x range]
			layers[d].ranges = np.zeros([layers[d-1].num_actions, layers[d-2].num_bets, layers[d-2].num_nonterminal_nonallin_nodes, batch_size, PC, HC], dtype=arguments.dtype)
			layers[d].cfvs = layers[d].ranges.copy()
			# self.lookahead.layers[d].placeholder_data = self.lookahead.layers[d].ranges.copy()
			layers[d].pot_size = np.full_like(layers[d].ranges, arguments.stack)
			# data structures [actions x parent_action x grandparent_id x batch x 1 x range]
			layers[d].strategies_avg = np.zeros([layers[d-1].num_actions, layers[d-2].num_bets, layers[d-2].num_nonterminal_nonallin_nodes, batch_size, HC], dtype=arguments.dtype)
			layers[d].current_strategy = layers[d].strategies_avg.copy()
			layers[d].regrets = np.full_like(layers[d].strategies_avg, self.lookahead.regret_epsilon)
			# self.lookahead.layers[d].current_regrets = np.zeros_like(self.lookahead.layers[d].strategies_avg)
			layers[d].empty_action_mask = np.ones_like(layers[d].strategies_avg)
		# create the optimized data structures for terminal equity
		self.lookahead.num_term_call_nodes = 0
		self.lookahead.num_term_fold_nodes = 0
		# calculate term_call_indices
		for d in range(1,self.lookahead.depth):
			if self.lookahead.tree.street != constants.streets_count:
				if d > 1 or self.lookahead.first_call_terminal:
					before = self.lookahead.num_term_call_nodes
					num_nonterminal_nonallin_grandparents = layers[d].ranges[1][-1].shape[0]
					self.lookahead.num_term_call_nodes += num_nonterminal_nonallin_grandparents
					layers[d].term_call_idx = np.array([before, self.lookahead.num_term_call_nodes], dtype=arguments.int_dtype)
			else:
				if d > 1 or self.lookahead.first_call_terminal:
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


	def reset(self):
		HC = constants.hand_count
		layers = self.lookahead.layers
		for d in range(0, self.lookahead.depth):
			if d in self.lookahead.layers:
				self.lookahead.layers[d].ranges.fill(1.0/HC)
				self.lookahead.layers[d].strategies_avg.fill(0)
				self.lookahead.layers[d].current_strategy.fill(0)
				self.lookahead.layers[d].cfvs.fill(0)
				self.lookahead.layers[d].cfvs_avg.fill(0)
				self.lookahead.layers[d].regrets.fill(0)
		if self.lookahead.next_street_boxes is not None:
			self.lookahead.next_street_boxes.iter = 0
			self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)


	def set_datastructures_from_tree_dfs(self, node, depth, action_id, parent_id, gp_id, cur_action_id, parent_action_id=None):
		''' Traverses the tree to fill in lookahead data structures that
			summarize data contained in the tree.
			 ex: saves pot sizes and numbers of actions at each lookahead state.
		@param: node the current node of the public tree
		@param: depth the depth of the current node
		@param: action_id the index of the action that led to this node
		@param: parent_id the index of the current node's parent
		@param: gp_id the index of the current node's grandparent
		'''
		# fill the potsize
		assert(node.pot)
		self.lookahead.layers[depth].pot_size[ action_id, parent_id, gp_id, : , : ] = node.pot
		if depth == 2 and cur_action_id == constants.actions.ccall:
			self.lookahead.parent_action_id[parent_id] = parent_action_id
		node.lookahead_coordinates = np.array([action_id, parent_id, gp_id], dtype=arguments.dtype)
		# transition call cannot be allin call
		if node.current_player == constants.players.chance:
			assert(parent_id <= self.lookahead.layers[depth-2].num_nonallin_bets)
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
					for child_id in range(num_terminal_actions):
						child_node = node.children[child_id]
						# go deeper
						self.set_datastructures_from_tree_dfs(child_node, depth+1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)
					# we need to make sure that even though there are fewer actions, the last action/allin is has the same last index as if we had full number of actions
					# we manually set the action_id as the last action (allin)
					for b in range(2, existing_num_bets-1):
						self.set_datastructures_from_tree_dfs(node.children[len(node.children)-b], depth+1, self.lookahead.layers[depth].num_actions-b, next_parent_id, next_gp_id,  node.actions[len(node.children)-b], cur_action_id)
					# mask out empty actions
					a = self.lookahead.layers[depth+1].empty_action_mask.shape[0] - existing_num_bets
					self.lookahead.layers[depth+1].empty_action_mask[ num_terminal_actions:a, next_parent_id, next_gp_id, : ] = 0
				else:
					# node has full action count, easy to handle
					for child_id in range(len(node.children)):
						child_node = node.children[child_id]
						# go deeper
						self.set_datastructures_from_tree_dfs(child_node, depth+1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)


	def build_from_tree(self, tree):
		''' Builds the lookahead's internal data structures using the public tree.
		@param: tree the public tree used to construct the lookahead
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
		# traverse the tree and fill the datastructures (pot sizes, non-existin actions, ...)
		# node, layer, action, parent_action, gp_id
		self.set_datastructures_from_tree_dfs(tree, depth=0, action_id=0, parent_id=0, gp_id=0, cur_action_id=-100)
		# we mask out fold as a possible action when check is for free, due to
		# 1) fewer actions means faster convergence
		# 2) we need to make sure prob of free fold is zero because ACPC dealer changes such action to check
		if self.lookahead.tree.bets[0] == self.lookahead.tree.bets[1]:
			self.lookahead.layers[1].empty_action_mask[0].fill(0)
		# construct the neural net query boxes
		self._construct_transition_boxes()


	def _compute_tree_structures(self, current_layer, current_depth):
		''' Computes the maximum number of actions at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
			The maximum number of actions is used so that every node at that
			depth can fit in the same tensor.
		@param: current_layer a list of tree nodes at the current depth
		@param: current_depth the depth of the current tree nodes
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
