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
		self.tree_builder = PokerTreeBuilder()
		self.verbose = verbose
		self.terminal_equity = terminal_equity
		self.lookahead = Lookahead(None, None)
		# temp bottom for lineprofiler
		# from helper_classes import Node
		# from Game.card_to_string_conversion import card_to_string
		# current_node = Node()
		# current_node.board = card_to_string.string_to_board('7d7c8s5s')
		# current_node.street = 3
		# current_node.current_player = constants.players.P2
		# current_node.bets = np.array([8000, 8000], dtype=arguments.dtype)
		# current_node.num_bets = 0
		# build_tree_params = TreeParams()
		# build_tree_params.root_node = current_node
		# build_tree_params.limit_to_street = True
		# self.lookahead_tree = self.tree_builder.build_tree(build_tree_params)
		# self.lookahead = Lookahead(self.terminal_equity, 1)
		# self.lookahead.build_lookahead(self.lookahead_tree)


	def _create_lookahead_tree(self, node):
		''' Builds a depth-limited public tree rooted at a given game node.
		@param: node the root of the tree
		'''
		build_tree_params = TreeParams()
		build_tree_params.root_node = node
		build_tree_params.limit_to_street = True
		self.lookahead_tree = self.tree_builder.build_tree(build_tree_params)


	def resolve(self, node, player_range, opponent_range=None, opponent_cfvs=None):
		if opponent_range is not None and opponent_cfvs is not None: raise('only 1 var can be passed')
		if opponent_range is None and opponent_cfvs is None: raise('one of those vars must be passed')
		batch_size = player_range.shape[0]
		if player_range.ndim == 1:
			player_range = player_range.reshape([1, player_range.shape[0]])
			opponent_range = opponent_range.reshape([1, opponent_range.shape[0]]) if opponent_range is not None else opponent_range
		# opponent_cfvs = None if we only need to resolve first node
		self._create_lookahead_tree(node)
		self.lookahead = Lookahead(self.terminal_equity, batch_size)
		if self.verbose > 0: t0 = time.time()
		self.lookahead.build_lookahead(self.lookahead_tree)
		if self.verbose > 0: print('Build time: {}'.format(time.time() - t0)); t0 = time.time()
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


	def _action_to_action_id(self, action):
		''' Gives the index of the given action at the node being re-solved.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action a legal action at the node
		@return the index of the action
		'''
		actions = self.get_possible_actions()
		action_id = -1
		for i in range(actions.shape[0]):
			if action == actions[i]:
				action_id = i
		assert(action_id != -1)
		return action_id


	def get_possible_actions(self):
		''' Gives a list of possible actions at the node being re-solved.
			 The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@return a list of legal actions
		'''
		return self.lookahead_tree.actions


	def get_root_cfv(self):
		''' Gives the average counterfactual values that the re-solve player
			received at the node during re-solving.
			The node must first be re-solved with @{resolve_first_node}.
		@return a vector of cfvs
		'''
		return self.resolve_results.root_cfvs


	def get_root_cfv_both_players(self):
		''' Gives the average counterfactual values that each player received
			at the node during re-solving.
			Usefull for data generation for neural net training
			The node must first be re-solved with @{resolve_first_node}.
		@return a (2,K) tensor of cfvs, where K is the range size
		'''
		return self.resolve_results.root_cfvs_both_players


	def get_action_cfv(self, action):
		''' Gives the average counterfactual values that the opponent received
			during re-solving after the re-solve player took a given action.
			Used during continual re-solving to track opponent cfvs. The node must
			first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action the action taken by the re-solve player
				at the node being re-solved
		@return a vector of cfvs
		'''
		action_id = self._action_to_action_id(action)
		return self.resolve_results.children_cfvs[action_id]


	def get_chance_action_cfv(self, action, board):
		''' Gives the average counterfactual values that the opponent received
			during re-solving after a chance event (the betting round changes and
			more cards are dealt).
			Used during continual re-solving to track opponent cfvs.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param: action the action taken by the re-solve player
				at the node being re-solved
		@param: board a vector of board cards
				which were updated by the chance event
		@return a vector of cfvs
		'''
		action_id = self._action_to_action_id(action)
		return self.lookahead.get_chance_action_cfv(action_id, board)


	def get_action_strategy(self, action):
		''' Gives the probability that the re-solved strategy takes a given action.
			The node must first be re-solved with @{resolve} or @{resolve_first_node}.
		@param action a legal action at the re-solve node
		@return a vector giving the probability of taking the action
				with each private hand
		'''
		action_id = self._action_to_action_id(action)
		return self.resolve_results.strategy[action_id]




#
