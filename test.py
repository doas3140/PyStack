from Tree.tree_builder import tree_builder
from Tree.tree_visualizer import tree_visualizer
from Tree.tree_values import TreeValues
from Tree.tree_cfr import TreeCFR
from Settings.constants import constants
from Settings.game_settings import game_settings
from Settings.arguments import arguments
from Game.card_tools import card_tools
from Game.card_to_string_conversion import card_to_string
from helper_classes import TreeParams, Node
import numpy as np
from time import time

PC, CC = constants.players_count, game_settings.card_count

params = TreeParams()

params.root_node = Node()
params.root_node.board = card_to_string.string_to_board('')
params.root_node.street = 1
params.root_node.current_player = constants.players.P1
params.root_node.bets = np.array([100, 100])

tree = tree_builder.build_tree(params)

starting_ranges = np.zeros([PC,CC], dtype=arguments.dtype)
starting_ranges[0] = card_tools.get_uniform_range(params.root_node.board)
starting_ranges[1] = card_tools.get_uniform_range(params.root_node.board)

t0 = time()
tree_cfr = TreeCFR()
tree_cfr.run_cfr(tree, starting_ranges)
print(time()-t0)

tree_values = TreeValues()
tree_values.compute_values(tree, starting_ranges)

print('Exploitability: ' + str(tree.exploitability) + ' [chips]' )

print(np.array2string(tree.strategy, suppress_small=True, precision=2))

# tree_visualizer.draw_tree(tree)
