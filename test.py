from Tree.tree_builder import tree_builder
from helper_classes import TreeParams

params = TreeParams()
params.init_root_node()

root = tree_builder.build_tree(params)
