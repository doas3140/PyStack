'''
    Generates visual representations of game trees.
'''
from graphviz import Digraph

from Settings.arguments import arguments
from Settings.constants import constants
# from Game.card_to_string_conversion import card_to_string

class TreeVisualiser():
    def __init__(self):
        self.i = 0
        self.g = None # graph

    def style_node(self, node, node_idx):
        node_type = node.node_type
        type = node.type
        terminal_str = ''
        chance_str = ''
        decision_str = str(node.current_player)
        # if node_type == constants.node_types.inner_node:
        #     print(node.type)
        if type == constants.node_types.terminal_fold:
            self.g.attr('node', shape='circle', color='red', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
            self.g.node(str(node_idx), terminal_str)
        if type == constants.node_types.check: # NOTE: check = terminal_call = -1
            self.g.attr('node', shape='circle', color='purple', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
            self.g.node(str(node_idx), decision_str)
        if type == constants.node_types.terminal_call:
            self.g.attr('node', shape='circle', color='blue', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
            self.g.node(str(node_idx), terminal_str)
        if node_type == constants.node_types.chance_node:
            self.g.attr('node', shape='circle', color='yellow', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
            self.g.node(str(node_idx), chance_str)
        if node_type == constants.node_types.inner_node:
            self.g.attr('node', shape='circle', color='orange', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
            self.g.node(str(node_idx), decision_str)
        self.g.attr('node', shape='circle', color='white', fontcolor='#03396c', fixedsize='true', width='0.3', height='0.3')
        self.g.node(str(node_idx), str(node.current_player))


    def style_edge(self, parent_action, parent_node, node, parent_idx, node_idx):
        if parent_node.node_type == constants.node_types.chance_node:
            edge_name = ''
        else:
            if parent_action == -1:
                edge_name = 'C'
            elif parent_action == -2:
                edge_name = 'F'
            else:
                edge_name = str(int(parent_action))
        self.g.edge(str(parent_idx), str(node_idx), edge_name)

    def dfs(self, parent, parent_index, depth=0):
        ''' recursively creates edges and nodes for graphviz '''
        for i, child in enumerate(parent.children):
            parent_action = parent.actions[i]
            self.i += 1
            self.style_node(child, self.i)
            self.style_edge(parent_action, parent, child, parent_index, self.i)
            self.dfs(child, str(self.i), depth+1)

    def draw_tree(self, root, name='tree', save_pdf=False, size='6,6'):
        self.i = 0
        self.g = Digraph(name, filename=name)
        self.g.attr(size=size, bgcolor='#fffef9')
        self.g.node_attr.update(style='filled', fontcolor='#e6e6ea')
        self.g.edge_attr.update(color='#e6e6ea', fontcolor='#03396c')
        # add first node (root node)
        self.style_node(root, self.i)
        # run depth first search and add edges and nodes to g
        self.dfs(root, parent_index=self.i, depth=0)
        # save pdf
        if save_pdf:
            self.g.view()
        return self.g


tree_visualizer = TreeVisualiser()




#
