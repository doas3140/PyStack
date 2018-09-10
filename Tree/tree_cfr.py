'''
	Runs Counterfactual Regret Minimization (CFR) to approximately
	solve a game represented by a complete game tree.
	As this class does full solving from the root of the game with no
	limited lookahead, it is not used in continual re-solving.
	It is provided simply for convenience.
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..Game.card_tools import card_tools
from ..TerminalEquity.terminal_equity import TerminalEquity

class TreeCFR():
	def __init__(self):
		''' for ease of implementation, we use small epsilon rather than
			zero when working with regrets
		'''
		pass


	def _get_terminal_equity(self, node):
		''' Gets an evaluator for player equities at a terminal node.
			Caches the result to minimize creation of
			@{terminal_equity|TerminalEquity} objects.
		@param: node the terminal node to evaluate
		@return a @{terminal_equity|TerminalEquity} evaluator for the node
		'''
		pass


	def cfrs_iter_dfs(self, node, iter):
		''' Recursively walks the tree, applying the CFR algorithm.
		@param: node the current node in the tree
		@param: iter the current iteration number
		'''
		pass
		# compute values using terminal_equity in terminal nodes
		# multiply by the pot
		# we have to compute current strategy at the beginning of each iteraton
		# initialize regrets in the first iteration
		# compute positive regrets so that we can compute the current strategy from them
		# compute the current strategy
		# current cfv [[actions, players, ranges]]
		# set new absolute ranges (after the action) for the child
		# computing regrets
		# accumulating average strategy


	def update_regrets(self, node, current_regrets):
		''' Update a node's total regrets with the current iteration regrets.
		@param: node the node to update
		@param: current_regrets the regrets from the current iteration of CFR
		'''
		pass


	def update_average_strategy(self, node, current_strategy, iter):
		''' Update a node's average strategy with the current iteration strategy.
		@param: node the node to update
		@param: current_strategy the CFR strategy for the current iteration
		@param: iter the iteration number of the current CFR iteration
		'''
		pass


	def run_cfr(self, root, starting_ranges, iter_count):
		''' Run CFR to solve the given game tree.
		@param root the root node of the tree to solve.
		@param: [opt] starting_ranges probability vectors over player private hands
				at the root node (default uniform)
		@param: [opt] iter_count the number of iterations to run CFR for
				(default @{arguments.cfr_iters})
		'''
		pass




#
