'''
	Implements depth-limited re-solving at a node of the game tree.
	Internally uses @{cfrd_gadget|CFRDGadget} TODO SOLVER
'''
import time
import numpy as np

from Lookahead.lookahead import Lookahead
from Lookahead.cfrd_gadget import CFRDGadget
from Tree.tree_builder import PokerTreeBuilder
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from helper_classes import TreeParams
from Tree.tree_values import TreeValues
from Tree.tree_cfr import TreeCFR


class Resolving():
	def __init__(self, terminal_equity, verbose=0):
		'''
		@param: TerminalEquity :object that evaluates rewards with specified board
		@param: int            :printing outputs if >0
		'''
		self.tree_builder = PokerTreeBuilder()
		self.verbose = verbose
		self.terminal_equity = terminal_equity


	def _create_lookahead_tree(self, node):
		''' Builds a depth-limited public tree rooted at a given game node
		@param: Node :root of the tree
		'''
		build_tree_params = TreeParams()
		build_tree_params.root_node = node
		build_tree_params.limit_to_street = True
		self.lookahead_tree = self.tree_builder.build_tree(build_tree_params)


	def resolve(self, node, player_range, opponent_range=None, opponent_cfvs=None):
		''' Creates lookahead and solves it
		@param: Node             :root node of the tree
		@param: [I]              :current player's range
		@param: [I]              :opponent's range
		@param: [I]              :opponent's cfvs (used to reconstruct opponent's range)
		@return LookaheadResults :results
		(only one of `opponent_range` and `opponent_cfvs` should be used)
		'''
		if opponent_range is not None:
			if opponent_range.ndim != 2: raise(Exception('opponent_range has to have batch size as first dim. (can be 1)'))
		if opponent_cfvs is not None:
			if opponent_cfvs.ndim != 1: raise(Exception('opponent_cfvs has to have only one dimension of 1326 numbers.'))
		if opponent_range is not None and opponent_cfvs is not None: raise(Exception('only 1 var can be passed'))
		if opponent_range is None and opponent_cfvs is None: raise(Exception('one of those vars must be passed'))
		# opponent_cfvs = None if we only need to resolve first node
		batch_size = player_range.shape[0]
		self._create_lookahead_tree(node)
		self.lookahead = Lookahead(self.lookahead_tree, self.terminal_equity, batch_size)
		if self.verbose > 0: t0 = time.time()
		if opponent_range is not None:
			self.lookahead.resolve(player_range=player_range, opponent_range=opponent_range)
			self.resolve_results = self.lookahead.get_results(reconstruct_opponent_cfvs=False)
		else: # opponent_cfvs is not None:
			self.lookahead.resolve(player_range=player_range, opponent_cfvs=opponent_cfvs)
			self.resolve_results = self.lookahead.get_results(reconstruct_opponent_cfvs=True)
		if self.verbose > 0: print('Resolve time: {}'.format(time.time() - t0))
		if self.verbose > 0:
			batch = 0
			print('printing batch:', batch)
			print('root_cfvs -', self.resolve_results.root_cfvs.shape)
			print(np.array2string(self.resolve_results.root_cfvs[batch].reshape([-1])[ 1320:1326 ], suppress_small=True, precision=2))
			print('root_cfvs_both_players -', self.resolve_results.root_cfvs_both_players.shape)
			print(np.array2string(self.resolve_results.root_cfvs_both_players[ : , 1-self.lookahead_tree.current_player , : ][batch].reshape([-1])[ 1320:1326 ], suppress_small=True, precision=2))
			print(np.array2string(self.resolve_results.root_cfvs_both_players[ : , self.lookahead_tree.current_player , : ][batch].reshape([-1])[ 1320:1326 ], suppress_small=True, precision=2))
			print('achieved_cfvs -', self.resolve_results.achieved_cfvs.shape)
			print(np.array2string(self.resolve_results.achieved_cfvs[batch].reshape([2,-1])[ 1-self.lookahead_tree.current_player , 1320:1326 ], suppress_small=True, precision=2))
			print(np.array2string(self.resolve_results.achieved_cfvs[batch].reshape([2,-1])[ self.lookahead_tree.current_player , 1320:1326 ], suppress_small=True, precision=2))
			print('strategy -', self.resolve_results.strategy.shape)
			a = self.resolve_results.strategy.shape[0]
			print(np.array2string(self.resolve_results.strategy[ : , batch, : ].reshape([a,-1])[ : , 1320:1326 ], suppress_small=True, precision=2))
		return self.resolve_results




#
