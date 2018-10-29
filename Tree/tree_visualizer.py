'''
    Generates visual representations of game trees.
'''
import numpy as np
from graphviz import Digraph

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string

class TreeVisualiser():
    def __init__(self):
        self.i = 0
        self.g = None # graph
        self.show_vars = None # bool
        self.C1 = '#d11141' # red
        self.C2 = '#00b159' # green
        self.C3 = '#00aedb' # blue
        self.C4 = '#f37735' # orange
        self.C5 = '#ffc425' # yellow
        self.BG = '#fffef9' #
        self.FT = '#fffef9'
        self.FTE = '#03396c' # edge

    def get_color(self, node):
        node_type = node.node_type
        type = node.type
        if type == constants.node_types.terminal_fold:
            return self.C1
        if type == constants.node_types.check:
            return self.C3 # not needed
        if type == constants.node_types.terminal_call:
            return self.C3
        if node_type == constants.node_types.chance_node:
            return self.C5
        if node_type == constants.node_types.inner_node:
            return self.C4
        else:
            return '#6f7c85'


    def style_node(self, node, node_idx):
        node_type = node.node_type
        type = node.type
        terminal_str = ''
        chance_str = ''
        decision_str = str(node.current_player)
        # if node_type == constants.node_types.inner_node:
        #     print(node.type)
        color = self.get_color(node)
        self.g.attr('node', shape='circle', color=color, fontcolor=self.FT, fixedsize='true', width='0.3', height='0.3')
        if type == constants.node_types.terminal_fold:
            self.g.node(str(node_idx), terminal_str)
        if type == constants.node_types.check: # NOTE: check = terminal_call = -1
            self.g.node(str(node_idx), decision_str)
        if type == constants.node_types.terminal_call:
            self.g.node(str(node_idx), terminal_str)
        if node_type == constants.node_types.chance_node:
            self.g.node(str(node_idx), chance_str)
        if node_type == constants.node_types.inner_node:
            self.g.node(str(node_idx), decision_str)
        self.g.node(str(node_idx), str(node.current_player))


    def style_node_with_vars(self, node, node_idx):
        color = self.get_color(node)
        self.g.attr('node', shape='box', color=color, fontcolor=self.FT, fontsize='10', fixedsize='true', width='1.5', height='0.5')
        bets = np.array2string(node.bets, suppress_small=True, precision=2)
        pot = np.array2string(node.pot, suppress_small=True, precision=2)
        board_str = card_to_string.cards_to_string(node.board)
        type = node.type
        if type == constants.node_types.terminal_fold:
            type = 'TF'
        elif type == constants.node_types.check:
            type = 'CC'
        elif type == constants.node_types.terminal_call:
            type = 'TC'
        type = type or node.node_type
        if type == constants.node_types.chance_node:
            type = 'C'
        elif type == constants.node_types.inner_node:
            type = 'D'
        depth = node.depth
        player = node.current_player + 1
        self.g.node(str(node_idx), 'B: {} T: {}\npot: {} board: "{}"\ndepth: {} P: {}'.format(bets, type, pot, board_str, depth, player))


    def style_edge_with_strats(self, parent_action, parent_node, node, parent_idx, node_idx):
        if parent_node.node_type == constants.node_types.chance_node:
            edge_name = ''
        else:
            if parent_node.actions[parent_action] == -1:
                edge_name = 'C'
            elif parent_node.actions[parent_action] == -2:
                edge_name = 'F'
            else: # == bet
                edge_name = str(int(parent_node.actions[parent_action]))
        if parent_node.strategy is None: strat = ''
        else: strat = np.array2string(parent_node.strategy[parent_action], suppress_small=True, precision=2)
        edge_name = '{} s:{}'.format(edge_name, strat)
        self.g.attr('edge', fontsize='6')
        self.g.edge(str(parent_idx), str(node_idx), edge_name)

    def style_edge(self, parent_action, parent_node, node, parent_idx, node_idx):
        if parent_node.node_type == constants.node_types.chance_node:
            edge_name = ''
        else:
            if parent_node.actions[parent_action] == -1:
                edge_name = 'C'
            elif parent_node.actions[parent_action] == -2:
                edge_name = 'F'
            else:
                edge_name = str(int(parent_node.actions[parent_action]))
        self.g.edge(str(parent_idx), str(node_idx), edge_name)

    def dfs(self, parent, parent_index, depth=0):
        ''' recursively creates edges and nodes for graphviz '''
        for i, child in enumerate(parent.children):
            self.i += 1
            if not self.show_vars: self.style_node(child, self.i)
            else: self.style_node_with_vars(child, self.i)
            if not self.show_vars: self.style_edge(i, parent, child, parent_index, self.i)
            else: self.style_edge_with_strats(i, parent, child, parent_index, self.i)
            self.dfs(child, str(self.i), depth+1)

    def draw_tree(self, root, name='tree', save_pdf=False, size='6,6', show_vars=False):
        self.show_vars = show_vars
        self.i = 0
        self.g = Digraph(name, filename=name)
        self.g.attr(size=size, bgcolor=self.BG)
        self.g.node_attr.update(style='filled', fontcolor=self.FT)
        self.g.edge_attr.update(color='#e6e6ea', fontcolor=self.FTE)
        # add first node (root node)
        if not self.show_vars: self.style_node(root, self.i)
        else: self.style_node_with_vars(root, self.i)
        # run depth first search and add edges and nodes to g
        self.dfs(root, parent_index=self.i, depth=0)
        # save pdf
        if save_pdf:
            self.g.view()
        return self.g


tree_visualizer = TreeVisualiser()




#
