'''
	Generates visual representations of game trees.
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Game.card_to_string_conversion import card_to_string

class TreeVisualiser():
	def __init__(self):
		self.node_to_graphviz_counter = 0
    	self.edge_to_graphviz_counter = 0


	def add_tensor(self, tensor, name, format, labels):
		''' Generates a string representation of a tensor.
		@param: tensor a tensor
		@param: [opt] name a name for the tensor
		@param: [opt] format a format string to use with @{string.format} for each
				element of the tensor
		@param: [opt] labels a list of labels for the elements of the tensor
		@return a string representation of the tensor
		'''
		pass


	def add_range_info(self, node):
		''' Generates a string representation of any range or value fields
			that are set for the given tree node.
		@param node the node
		@return a string containing concatenated representations of any tensors
				stored in the `ranges_absolute`, `cf_values`, or `cf_values_br`
				fields of the node.
		'''
		pass
		# cf values computed by real tree dfs
		# cf values that br has in real tree


	def node_to_graphviz(self, node):
		''' Generates data for a graphical representation of a node
			in a public tree.
		@param: node the node to generate data for
		@return a table containing `name`, `label`, and `shape` fields for graphviz
		'''
		pass
		# 1.0 label
		# 2.0 name
		# 3.0 shape


	def nodes_to_graphviz_edge(self, from, to, node, child_node):
		''' Generates data for graphical representation of a public tree action
			as an edge in a tree.
		@param: from the graphical node the edge comes from
		@param: to the graphical node the edge goes to
		@param: node the public tree node before at which the action is taken
		@param: child_node the public tree node that results from taking the action
		@return a table containing fields `id_from`, `id_to`, `id` for graphviz and
 				a `strategy` field to use as a label for the edge
		'''
		pass
		# get the child id of the child node


	def graphviz_dfs(self, node, nodes, edges):
		''' Recursively generates graphviz data from a public tree.
		@param node the current node in the public tree
		@param nodes a table of graphical nodes generated so far
		@param edges a table of graphical edges generated so far
		'''
		pass


	def graphviz(self, root, filename):
		''' Generates `.dot` and `.svg` image files which graphically represent
			a game's public tree.
			Each node in the image lists the acting player,
			the number of chips committed by each player, the current betting round,
			public cards and the depth of the subtree after the node,
			as well as any probabilities or values stored in the
			`ranges_absolute`, `cf_values`, or `cf_values_br` fields of the node.

			Each edge in the image lists the probability of the action being taken
			with each private card.
		@param: root the root of the game's public tree
		@param: filename a name used for the output files
		'''
		pass
		# write into dot file
		# run graphviz program to generate image




#
