'''
	Builds the internal data structures of a @{lookahead|Lookahead} object.
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
# from Tree.tree_builder import PokerTreeBuilder
# from Tree.tree_visualiser import TreeVisualiser
from Nn.next_round_value import NextRoundValue
from Nn.value_nn import ValueNn
from helper_classes import LookaheadLayer

NEURAL_NET = {}
AUX_NET = None
NEXT_ROUND_PRE = None

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
		# load neural net if not already loaded
		if self.lookahead.tree.street in NEURAL_NET:
			nn = NEURAL_NET[self.lookahead.tree.street]
		else:
			nn = ValueNn(pretrained_weights=True, verbose=0)
		NEURAL_NET[self.lookahead.tree.street] = nn

		if self.lookahead.tree.street == 1 and game_settings.nl:
			if AUX_NET is None:
				aux_net = ValueNn(self.lookahead.tree.street, True)
				AUX_NET = aux_net
			else:
				aux_net = AUX_NET

		self.lookahead.next_street_boxes = None
		self.lookahead.indices = {}
		self.lookahead.num_pot_sizes = 0

		if self.lookahead.tree.street == 1:
			if NEXT_ROUND_PRE is None:
				self.lookahead.next_street_boxes = NEXT_ROUND_PRE
			else:
				self.lookahead.next_street_boxes = NextRoundValuePre(nn, aux_net, self.lookahead.terminal_equity.board)
			NEXT_ROUND_PRE = self.lookahead.next_street_boxes
		else:
			self.lookahead.next_street_boxes = NextRoundValue(nn, self.lookahead.terminal_equity.board)

		# create the optimized data structures for batching next_round_value

		for d in range(1,self.lookahead.depth):
			if d == 1 and self.lookahead.first_call_transition:
				before = self.lookahead.num_pot_sizes
				self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + 1
				self.lookahead.layers[d].indices = np.array([before, self.lookahead.num_pot_sizes])
			elif not game_settings.nl and (d > 1 or self.lookahead.first_call_transition):
				before = self.lookahead.num_pot_sizes
				self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.layers[d].pot_size[1].shape[0]) * self.lookahead.layers[d].pot_size[1].shape[1]
				self.lookahead.layers[d].indices = np.array([before, self.lookahead.num_pot_sizes])
			elif self.lookahead.layers[d].pot_size[1].shape[0] > 1:
				before = self.lookahead.num_pot_sizes
				self.lookahead.num_pot_sizes = self.lookahead.num_pot_sizes + (self.lookahead.layers[d].pot_size[1].shape[0] - 1) * self.lookahead.layers[d].pot_size[1].shape[1]
				self.lookahead.layers[d].indices = np.array([before, self.lookahead.num_pot_sizes])

		if self.lookahead.num_pot_sizes == 0:
			return

		self.lookahead.next_round_pot_sizes = np.zeros([self.lookahead.num_pot_sizes], dtype=arguments.dtype)

		self.lookahead.action_to_index = {}
		for d in range(1,self.lookahead.depth):
			p_start, p_end = 0, -1 # parent_indices
			if d in self.lookahead.indices:
				if d == 1:
					p_start, p_end = 0, 1 # parent_indices
				elif not game_settings.nl:
					p_start, p_end = 0, self.layers[d].pot_size.shape[1] # parent indices
				self.lookahead.next_round_pot_sizes[ self.lookahead.layers[d].indices[0]:self.lookahead.layers[d].indices[1] ] = self.lookahead.layers[d].pot_size[ 1, p_Start:p_end, : , 0, 0, 0 ].copy()
				if d <= 2:
					if d == 1:
						assert(self.lookahead.layers[d].indices[0] == self.lookahead.layers[d].indices[1])
						self.lookahead.action_to_index[constants.actions.ccall] = self.lookahead.layers[d].indices[0]
					else:
						assert(self.lookahead.layers[d].pot_size[1, p_Start:p_end].shape[1] == 1) # bad num_indices
						for parent_action_idx in range(0, self.lookahead.layers[d].pot_size[1].shape[0]):
							action_id = self.lookahead.parent_action_id[parent_action_idx]
							assert(action_id not in self.lookahead.action_to_index)
							self.lookahead.action_to_index[action_id] = self.lookahead.layers[d].indices[0] + parent_action_idx - 1

		if constants.actions.ccall not in self.lookahead.action_to_index:
			print(self.lookahead.action_to_index)
			print(self.lookahead.parent_action_id)
			assert(False)

		PC, HC = constants.players_count, game_settings.hand_count
		self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)
		self.lookahead.next_street_boxes_inputs = np.zeros([self.lookahead.num_pot_sizes, self.lookahead.batch_size, PC, HC], dtype=arguments.dtype)
		self.lookahead.next_street_boxes_outputs = self.lookahead.next_street_boxes_inputs.copy()


	def _compute_structure(self):
		''' Computes the number of nodes at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
		'''
		assert(self.lookahead.tree.street >= 1 and self.lookahead.tree.street <= constants.streets_count)
		self.lookahead.regret_epsilon = 1.0 / 1000000000
		# which player acts at particular depth
		self.lookahead.layers[0].acting_player = 0
		for d in range(1, self.lookahead.depth+1):
			self.lookahead.layers[d].acting_player = 1 - self.lookahead.layers[d-1].acting_player
		# compute the node counts
		self.lookahead.layers[0].nonterminal_nodes_count = 1
		self.lookahead.layers[1].nonterminal_nodes_count = self.lookahead.layers[0].bets_count
		# self.lookahead.nonterminal_nonallin_nodes_count[0] = 1
		self.lookahead.layers[0].nonterminal_nonallin_nodes_count = 1
		self.lookahead.layers[1].nonterminal_nonallin_nodes_count = self.lookahead.layers[1].nonterminal_nodes_count - 1 if game_settings.nl else self.lookahead.layers[1].nonterminal_nodes_count
		self.lookahead.layers[0].all_nodes_count = 1
		self.lookahead.layers[1].all_nodes_count = self.lookahead.layers[0].actions_count
		self.lookahead.layers[0].allin_nodes_count = 0
		self.lookahead.layers[1].allin_nodes_count = 1
		# neural network input and output boxes
		self.lookahead.next_street_boxes_inputs = {}
		self.lookahead.next_street_boxes_outputs = {}
		for d in range(1, self.lookahead.depth):
			self.lookahead.layers[d+1].all_nodes_count = self.lookahead.layers[d-1].nonterminal_nonallin_nodes_count * self.lookahead.layers[d-1].bets_count * self.lookahead.layers[d].actions_count
			self.lookahead.layers[d+1].allin_nodes_count = self.lookahead.layers[d-1].nonterminal_nonallin_nodes_count * self.lookahead.layers[d-1].bets_count * 1
			self.lookahead.layers[d+1].nonterminal_nodes_count = self.lookahead.layers[d-1].nonterminal_nonallin_nodes_count * self.lookahead.layers[d-1].nonallinbets_count * self.lookahead.layers[d].bets_count
			self.lookahead.layers[d+1].nonterminal_nonallin_nodes_count = self.lookahead.layers[d-1].nonterminal_nonallin_nodes_count * self.lookahead.layers[d-1].nonallinbets_count * self.lookahead.layers[d].nonallinbets_count
			self.lookahead.layers[d].next_street_boxes_inputs = None
			self.lookahead.layers[d].next_street_boxes_outputs = None


	def construct_data_structures(self):
		''' Builds the tensors that store lookahead data during re-solving.
		'''
		PC, HC, batch_size = constants.players_count, game_settings.hand_count, self.lookahead.batch_size
		self._compute_structure()
		# lookahead main data structures
		# all the structures are per-layer tensors, that is, each layer holds the data in n-dimensional tensors
		# create the data structure for the first two layers
		# data structures [actions x parent_action x grandparent_id x batch x players x range]
		self.lookahead.layers[0].ranges_data = np.full([1, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		self.lookahead.layers[1].ranges_data = np.full([self.lookahead.layers[0].actions_count, 1, 1, batch_size, PC, HC], 1.0/HC, dtype=arguments.dtype)
		self.lookahead.layers[0].pot_size = np.zeros_like(self.lookahead.layers[0].ranges_data)
		self.lookahead.layers[1].pot_size = np.zeros_like(self.lookahead.layers[1].ranges_data)
		self.lookahead.layers[0].cfvs_data = np.zeros_like(self.lookahead.layers[0].ranges_data)
		self.lookahead.layers[1].cfvs_data = np.zeros_like(self.lookahead.layers[1].ranges_data)
		self.lookahead.layers[0].average_cfvs_data = np.zeros_like(self.lookahead.layers[0].ranges_data)
		self.lookahead.layers[1].average_cfvs_data = np.zeros_like(self.lookahead.layers[1].ranges_data)
		self.lookahead.layers[0].placeholder_data = np.zeros_like(self.lookahead.layers[0].ranges_data)
		self.lookahead.layers[1].placeholder_data = np.zeros_like(self.lookahead.layers[1].ranges_data)
		# data structures for one player [actions x parent_action x grandparent_id x 1 x range]
		self.lookahead.layers[0].average_strategies_data = None
		self.lookahead.layers[1].average_strategies_data = np.zeros([self.lookahead.layers[0].actions_count, 1, 1, batch_size, HC], dtype=arguments.dtype)
		self.lookahead.layers[0].current_strategy_data = None
		self.lookahead.layers[1].current_strategy_data = np.zeros_like(self.lookahead.layers[1].average_strategies_data)
		self.lookahead.layers[0].regrets_data = None
		self.lookahead.layers[1].regrets_data = np.zeros_like(self.lookahead.layers[1].average_strategies_data)
		self.lookahead.layers[0].current_regrets_data = None
		self.lookahead.layers[1].current_regrets_data = np.zeros_like(self.lookahead.layers[1].average_strategies_data)
		self.lookahead.layers[0].positive_regrets_data = None
		self.lookahead.layers[1].positive_regrets_data = np.zeros_like(self.lookahead.layers[1].average_strategies_data)
		self.lookahead.layers[0].empty_action_mask = None
		self.lookahead.layers[1].empty_action_mask = np.ones_like(self.lookahead.layers[1].average_strategies_data)
		# data structures for summing over the actions [1 x parent_action x grandparent_id x range]
		self.lookahead.layers[0].regrets_sum = np.zeros([1, 1, 1, batch_size, HC], dtype=arguments.dtype)
		self.lookahead.layers[1].regrets_sum = np.zeros([1, self.lookahead.layers[0].bets_count, 1, batch_size, HC], dtype=arguments.dtype)
		# data structures for inner nodes (not terminal nor allin) [bets_count x parent_nonallinbetscount x gp_id x batch x players x range]
		self.lookahead.layers[0].inner_nodes = np.zeros([1, 1, 1, batch_size, PC, HC], dtype=arguments.dtype)
		self.lookahead.layers[0].swap_data = np.transpose(self.lookahead.layers[0].inner_nodes, [0,2,1,3,4,5]) # :transpose(2,3):clone()
		self.lookahead.layers[0].inner_nodes_p1 = np.zeros([1, 1, 1, batch_size, 1, HC], dtype=arguments.dtype)
		if self.lookahead.depth > 2:
			self.lookahead.layers[1].inner_nodes = np.zeros([self.lookahead.layers[0].bets_count, 1, 1, batch_size, PC, HC], dtype=arguments.dtype)
			self.lookahead.layers[1].swap_data = np.transpose(self.lookahead.layers[1].inner_nodes, [0,2,1,3,4,5]) # :transpose(2,3):clone()
			self.lookahead.layers[1].inner_nodes_p1 = np.zeros([self.lookahead.layers[0].bets_count, 1, 1, batch_size, 1, HC], dtype=arguments.dtype)
		# create the data structures for the rest of the layers
		for d in range(2, self.lookahead.depth):
			# data structures [actions x parent_action x grandparent_id x batch x players x range]
			self.lookahead.layers[d].ranges_data = np.zeros([self.lookahead.layers[d-1].actions_count, self.lookahead.layers[d-2].bets_count, self.lookahead.layers[d-2].nonterminal_nonallin_nodes_count, batch_size, PC, HC], dtype=arguments.dtype)
			self.lookahead.layers[d].cfvs_data = self.lookahead.layers[d].ranges_data.copy()
			self.lookahead.layers[d].placeholder_data = self.lookahead.layers[d].ranges_data.copy()
			self.lookahead.layers[d].pot_size = np.full_like(self.lookahead.layers[d].ranges_data, arguments.stack)
			# data structures [actions x parent_action x grandparent_id x batch x 1 x range]
			self.lookahead.layers[d].average_strategies_data = np.zeros([self.lookahead.layers[d-1].actions_count, self.lookahead.layers[d-2].bets_count, self.lookahead.layers[d-2].nonterminal_nonallin_nodes_count, batch_size, HC], dtype=arguments.dtype)
			self.lookahead.layers[d].current_strategy_data = self.lookahead.layers[d].average_strategies_data.copy()
			self.lookahead.layers[d].regrets_data = np.full_like(self.lookahead.layers[d].average_strategies_data, self.lookahead.regret_epsilon)
			self.lookahead.layers[d].current_regrets_data = np.zeros_like(self.lookahead.layers[d].average_strategies_data)
			self.lookahead.layers[d].empty_action_mask = np.ones_like(self.lookahead.layers[d].average_strategies_data)
			self.lookahead.layers[d].positive_regrets_data = self.lookahead.layers[d].regrets_data.copy()
			# data structures [1 x parent_action x grandparent_id x batch x players x range]
			self.lookahead.layers[d].regrets_sum = np.zeros([1, self.lookahead.layers[d-2].bets_count, self.lookahead.layers[d-2].nonterminal_nonallin_nodes_count, batch_size, HC], dtype=arguments.dtype)
			# data structures for the layers except the last one
			if d < self.lookahead.depth:
				self.lookahead.layers[d].inner_nodes = np.zeros([self.lookahead.layers[d-1].bets_count, self.lookahead.layers[d-2].nonallinbets_count, self.lookahead.layers[d-2].nonterminal_nonallin_nodes_count, batch_size, PC, HC], dtype=arguments.dtype)
				self.lookahead.layers[d].inner_nodes_p1 = np.zeros([self.lookahead.layers[d-1].bets_count, self.lookahead.layers[d-2].nonallinbets_count, self.lookahead.layers[d-2].nonterminal_nonallin_nodes_count, batch_size, 1, HC], dtype=arguments.dtype)
				self.lookahead.layers[d].swap_data = np.transpose(self.lookahead.layers[d].inner_nodes, [0,2,1,3,4,5]) # :transpose(2, 3):clone()
		# create the optimized data structures for terminal equity
		self.lookahead.term_call_indices = {}
		self.lookahead.num_term_call_nodes = 0
		self.lookahead.term_fold_indices = {}
		self.lookahead.num_term_fold_nodes = 0
		# calculate term_call_indices
		for d in range(1,self.lookahead.depth):
			if self.lookahead.tree.street != constants.streets_count:
				if game_settings.nl and (d>1 or self.lookahead.first_call_terminal):
					before = self.lookahead.num_term_call_nodes
					self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.layers[d].ranges_data[1][-1].shape[0]
					self.lookahead.layers[d].term_call_indices = np.array([before, self.lookahead.num_term_call_nodes])
			else:
				if d>1 or self.lookahead.first_call_terminal:
					before = self.lookahead.num_term_call_nodes
					self.lookahead.num_term_call_nodes = self.lookahead.num_term_call_nodes + self.lookahead.layers[d].ranges_data[1].shape[0] * self.lookahead.layers[d].ranges_data[1].shape[1]
					self.lookahead.layers[d].term_call_indices = np.array([before, self.lookahead.num_term_call_nodes])
		# calculate term_fold_indices
		for d in range(1,self.lookahead.depth):
			before = self.lookahead.num_term_fold_nodes
			self.lookahead.num_term_fold_nodes = self.lookahead.num_term_fold_nodes + self.lookahead.layers[d].ranges_data[0].shape[0] * self.lookahead.layers[d].ranges_data[0].shape[1]
			self.lookahead.layers[d].term_fold_indices = np.array([before, self.lookahead.num_term_fold_nodes])
		self.lookahead.ranges_data_call = np.zeros([self.lookahead.num_term_call_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		self.lookahead.ranges_data_fold = np.zeros([self.lookahead.num_term_fold_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		self.lookahead.cfvs_data_call = np.zeros([self.lookahead.num_term_call_nodes, batch_size, PC, HC], dtype=arguments.dtype)
		self.lookahead.cfvs_data_fold = np.zeros([self.lookahead.num_term_fold_nodes, batch_size, PC, HC], dtype=arguments.dtype)


	def reset(self):
		HC = game_settings.hand_count
		for d in range(0, self.lookahead.depth):
			if d in self.lookahead.ranges_data:
				self.lookahead.layers[d].ranges_data.fill(1.0/HC)
			if d in self.lookahead.average_strategies_data:
				self.lookahead.layers[d].average_strategies_data.fill(0)
			if d in self.lookahead.current_strategy_data:
				self.lookahead.layers[d].current_strategy_data.fill(0)
			if d in self.lookahead.cfvs_data:
				self.lookahead.layers[d].cfvs_data.fill(0)
			if d in self.lookahead.average_cfvs_data:
				self.lookahead.layers[d].average_cfvs_data.fill(0)
			if d in self.lookahead.regrets_data:
				self.lookahead.layers[d].regrets_data.fill(0)
			if d in self.lookahead.current_regrets_data:
				self.lookahead.layers[d].current_regrets_data.fill(0)
			if d in self.lookahead.positive_regrets_data:
				self.lookahead.layers[d].positive_regrets_data.fill(0)
			if d in self.lookahead.placeholder_data:
				self.lookahead.layers[d].placeholder_data.fill(0)
			if d in self.lookahead.regrets_sum:
				self.lookahead.layers[d].regrets_sum.fill(0)
			if d in self.lookahead.inner_nodes:
				self.lookahead.layers[d].inner_nodes.fill(0)
			if d in self.lookahead.inner_nodes_p1:
				self.lookahead.layers[d].inner_nodes_p1.fill(0)
			if d in self.lookahead.swap_data:
				self.lookahead.layers[d].swap_data.fill(0)
		if self.lookahead.next_street_boxes is not None:
			self.lookahead.next_street_boxes.iter = 0
			self.lookahead.next_street_boxes.start_computation(self.lookahead.next_round_pot_sizes, self.lookahead.batch_size)


	def set_datastructures_from_tree_dfs(self, node, depth, action_id, parent_id, gp_id, cur_action_id, parent_action_id):
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
			assert(parent_id <= self.lookahead.layers[depth-2].nonallinbets_count)
		if depth < self.lookahead.depth + 1:
			gp_num_nonallin_bets = self.lookahead.layers[depth-2].nonallinbets_count if depth > 1 else 1
			p_num_terminal_actions = self.lookahead.layers[depth-1].terminal_actions_count if depth > 0 else 0
			# compute next coordinates for parent and grandparent
			next_parent_id = action_id - p_num_terminal_actions
			next_gp_id = gp_id * gp_num_nonallin_bets + parent_id
			if (not node.terminal) and (node.current_player != constants.players.chance):
				# parent is not an allin raise
				if depth > 1:
					assert(parent_id <= self.lookahead.layers[depth-2].nonallinbets_count)
				# do we need to mask some actions for that node? (that is, does the node have fewer children than the max number of children for any node on this layer)
				if len(node.children) < self.lookahead.layers[depth].actions_count:
					# we need to mask nonexisting padded bets
					assert(depth > 0)
					terminal_actions_count = self.lookahead.layers[depth].terminal_actions_count
					assert(terminal_actions_count == 2)
					existing_bets_count = len(node.children) - terminal_actions_count
					# allin situations
					if existing_bets_count == 0:
						if depth > 0:
							assert(action_id == self.lookahead.layers[depth-1].actions_count-1)
					for child_id in range(terminal_actions_count):
						child_node = node.children[child_id]
						# go deeper
						self.set_datastructures_from_tree_dfs(child_node, depth+1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id)
					# we need to make sure that even though there are fewer actions, the last action/allin is has the same last index as if we had full number of actions
					# we manually set the action_id as the last action (allin)
					for b in range(0, existing_bets_count):
						self.set_datastructures_from_tree_dfs(node.children[len(node.children)-b], depth+1, self.lookahead.layers[depth].actions_count-b, next_parent_id, next_gp_id,  node.actions[len(node.children)-b], cur_action_id)
					# mask out empty actions
					a = self.lookahead.layers[depth+1].empty_action_mask.shape[0] - existing_bets_count
					self.lookahead.layers[depth+1].empty_action_mask[ terminal_actions_count:a, next_parent_id, next_gp_id, : ] = 0
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
		self.construct_data_structures()
		# action ids for first
		self.lookahead.parent_action_id = {}
		# traverse the tree and fill the datastructures (pot sizes, non-existin actions, ...)
		# node, layer, action, parent_action, gp_id
		self.set_datastructures_from_tree_dfs(tree, 0, 0, 0, 0, 0, -100)
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
		layer_actions_count, layer_terminal_actions_count = 0, 0
		next_layer = []
		for n in range(len(current_layer)):
			node = current_layer[n]
			layer_actions_count = max(layer_actions_count, len(node.children))
			node_terminal_actions_count = 0
			for c in range(len(current_layer[n].children)):
				if node.children[c].terminal or node.children[c].current_player == constants.players.chance:
					node_terminal_actions_count += 1
			layer_terminal_actions_count = max(layer_terminal_actions_count, node_terminal_actions_count)
			# add children of the node to the next layer for later pass of BFS
			if not node.terminal:
				for c in range(len(node.children)):
					next_layer.append(node.children[c])
		assert((layer_actions_count == 0) == (len(next_layer) == 0))
		assert((layer_actions_count == 0) == (current_depth == self.lookahead.depth - 1))
		# set action and bet counts
		self.lookahead.layers[current_depth].bets_count = layer_actions_count - layer_terminal_actions_count
		if game_settings.nl: self.lookahead.layers[current_depth].nonallinbets_count = self.lookahead.layers[current_depth].bets_count - 1 # remove allin
		else: self.lookahead.layers[current_depth].nonallinbets_count = self.lookahead.layers[current_depth].bets_count
		# if no allin...
		if layer_actions_count == 2:
			assert(layer_actions_count == layer_terminal_actions_count)
			self.lookahead.layers[current_depth].nonallinbets_count = 0
		self.lookahead.layers[current_depth].terminal_actions_count = layer_terminal_actions_count
		self.lookahead.layers[current_depth].actions_count = layer_actions_count
		if len(next_layer) > 0:
			assert(layer_actions_count >= 2)
			# go deeper
			self._compute_tree_structures(next_layer, current_depth + 1)




#
