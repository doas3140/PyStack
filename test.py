from Tree.tree_builder import tree_builder
from Tree.tree_visualizer import tree_visualizer
from Tree.tree_values import TreeValues
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from helper_classes import TreeParams, Node
import numpy as np


params = TreeParams()

params.root_node = Node()
params.root_node.board = card_to_string.string_to_board('')
params.root_node.street = 1
params.root_node.current_player = constants.players.P1
params.root_node.bets = np.array([100, 100])

tree = tree_builder.build_tree(params)

tree_values = TreeValues()
tree_values.compute_values(tree)

print('Exploitability: ' + str(tree.exploitability) + ' [chips]' )

tree_visualizer.draw_tree(tree)
