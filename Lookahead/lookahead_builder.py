'''
	Builds the internal data structures of a @{lookahead|Lookahead} object.
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..tools import tools
# from ..Tree.tree_builder import PokerTreeBuilder # dont need here?
# from ..Tree.tree_visualiser import TreeVisualiser # dont need here?
from ..Nn.next_round_value import NextRoundValue
from ..Nn.value_nn import ValueNn


class LookaheadBuilder():
	def __init__(self, lookahead):
		pass

	def _construct_transition_boxes(self):
		''' Builds the neural net query boxes which estimate
			counterfactual values at depth-limited states of the lookahead.
		'''
		pass


	def _compute_structure(self):
		''' Computes the number of nodes at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
		'''
		pass


	def construct_data_structures(self):
		''' Builds the tensors that store lookahead data during re-solving.
		'''
		pass


	def set_datastructures_from_tree_dfs(self, node, layer, action_id, parent_id, gp_id):
		''' Traverses the tree to fill in lookahead data structures that
			summarize data contained in the tree.
 			ex: saves pot sizes and numbers of actions at each lookahead state.
		@param: node the current node of the public tree
		@param: layer the depth of the current node
		@param: action_id the index of the action that led to this node
		@param: parent_id the index of the current node's parent
		@param: gp_id the index of the current node's grandparent
		'''
		pass


	def build_from_tree(self, tree):
		''' Builds the lookahead's internal data structures using the public tree.
		@param: tree the public tree used to construct the lookahead
		'''
		pass


	def _compute_tree_structures(self, current_layer, current_depth):
		''' Computes the maximum number of actions at each depth of the tree.
			Used to find the size for the tensors which store lookahead data.
			The maximum number of actions is used so that every node at that
			depth can fit in the same tensor.
		@param: current_layer a list of tree nodes at the current depth
		@param: current_depth the depth of the current tree nodes
		'''
		pass




#
