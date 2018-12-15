'''
	Performs the main steps of continual re-solving, tracking player range
	and opponent counterfactual values so that re-solving can be done at each
	new game state.
'''
import numpy as np

from Game.card_to_string_conversion import card_to_string
from TerminalEquity.terminal_equity import TerminalEquity
from Lookahead.resolving import Resolving
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools
from helper_classes import Node

class ContinualResolving():
	def __init__(self, verbose=1):
		''' Does a depth-limited solve of the game's first node '''
		HC = constants.hand_count
		self.verbose = verbose
		self.uniform_range = np.full([1,HC], 1/HC, dtype=arguments.dtype)
		self.terminal_equity = TerminalEquity()
		self._cache_initial_opponent_cfvs()


	def start_new_hand(self, card1, card2, player_is_small_blind):
		''' Re-initializes the continual re-solving to start a new game
			from the root of the game tree
		@param: card: string of 2 chars. first is rank (if letter then only capital) and second suit (lower case).
				ex: '2c', '6d', 'Jh', 'Ks', 'Ad'. note: for 10 use 'Ts' not '10s'
		'''
		card1, card2 = card_to_string.string_to_card(card1), card_to_string.string_to_card(card2)
		P1, P2 = constants.players.P1, constants.players.P2
		self.prev_street = -1
		self.prev_bet = None
		self.player_position = P1 if player_is_small_blind else P2
		self.hand_id = card_tools.get_hole_index([card1, card2])
		# init player range and opponent cfvs
		self.player_range = self.uniform_range.copy()
		start_cfvs = self.starting_cfvs_as_P1 if self.player_position == P2 else self.starting_cfvs_as_P2
		self.opponent_cfvs = start_cfvs.copy()


	def compute_action(self, board_string, player_bet, opponent_bet):
		''' Re-solves a node and chooses the re-solving player's next action '''
		# create node
		node = self._create_node(board_string, player_bet, opponent_bet)
		# if street changed (last node was chance node) and it is not first action, then update cfvs and ranges
		if self.prev_street+1 == node.street and self.prev_street == -1:
			assert(not node.terminal)
			# opponent cfvs: if the street has changed, the resonstruction API simply gives us CFVs
			self.opponent_cfvs = self.resolving.get_chance_action_cfv(self.prev_action, node.board)
			# player range: if street has change, we have to mask out the colliding hands
			mask = self.get_possible_hand_indexes(node.board)
			self.player_range *= mask						# mask available combinations given particular board
			self.player_range /= self.player_range.sum()	# normalize
			# set terminal equity for new board
			self.terminal_equity.set_board(node.board)
		# elif last node was opponents turn: do nothing
		# resolve and sample bet
		self.resolving = Resolving(self.terminal_equity)
		results = self.resolving.resolve(node, self.player_range, opponent_cfvs=self.opponent_cfvs)
		if self.verbose != 0:
			for card1 in range(52):
				for card2 in range(card1+1, 52):
					idx = card_tools.get_hole_index([card1,card2])
					c1, c2 = card_to_string.card_to_string(card1), card_to_string.card_to_string(card2)
					print(c1, c2, np.array2string(results.strategy[:,0,idx], suppress_small=True, precision=2))
		# sample bet
		strategy = results.strategy[ : , 0 , self.hand_id ] # [A,b,I] -> [A], here b = 1
		assert(abs(1 - strategy.sum()) < 0.001)
		action_idx = np.random.choice(np.arange(len(strategy)), p=strategy)
		possible_bets = self.resolving.get_possible_actions()
		sampled_bet = possible_bets[action_idx]
		print( "strat: {}, bets: {}, sampled_bet: {}".format(strategy, possible_bets, sampled_bet) )
		# update the invariants based on our action # [I] = [I]
		self.opponent_cfvs = results.children_cfvs[action_idx,0,:] # [A,b,I], here b = 1
		# [I] *= [I]
		self.player_range *= results.strategy[action_idx,0,:] # [A,b,I], here b = 1
		self.player_range /= self.player_range.sum()	# normalize
		# update history variables
		self.prev_action = action_idx
		self.prev_street = node.street
		# return action
		if sampled_bet == constants.actions.fold:
			return {'action':'fold', 'amount': -1}
		elif sampled_bet == constants.actions.ccall:
			return {'action':'call', 'amount': -1}
		else:
			return {'action':'raise', 'amount': sampled_bet}


	def _create_node(self, board_string, player_bet, opponent_bet):
		P1, P2 = constants.players.P1, constants.players.P2
		node = Node()
		node.board = card_to_string.string_to_board(board_string)
		node.street = card_tools.board_to_street(node.board)
		node.current_player = self.player_position
		# note: P1 is always playing small blind, P2 - big blind, but players are sometimes swaped
		P1_bet, P2_bet = (player_bet,opponent_bet) if self.player_position == P1 else (opponent_bet,player_bet)
		node.bets = np.array([P1_bet, P2_bet], dtype=arguments.dtype)
		# bets_are_initial = (P1_bet == sb and P2_bet == bb) or (P1_bet == bb and P2_bet == sb)
		node.num_bets = 1 if self.prev_street == -1 else 0
		return node


	def _cache_initial_opponent_cfvs(self):
		''' Solves a depth-limited lookahead from the first node of the game
			to get opponent counterfactual values. The cfvs are returned.
			Because this is the first node of the game,
			exact ranges are known for both players,
			so opponent cfvs are not necessary for solving.
			Save in `starting_cfvs_as_P1` and `starting_cfvs_as_P2`
		'''
		P1, P2 = constants.players.P1, constants.players.P2
		self.player_position = P1 # set it temporary (used in _create_node)
		self.prev_street = -1  # set it to beggining (used in _create_node)
		first_node = self._create_node(board_string='', player_bet=arguments.sb, opponent_bet=arguments.bb)
		# set board
		self.terminal_equity.set_board(first_node.board)
		# create re-solving and re-solve the first node
		self.first_node_resolving = Resolving(self.terminal_equity)
		results = self.first_node_resolving.resolve(first_node, player_range=self.uniform_range, opponent_range=self.uniform_range)
		# returns the initial CFVs [b, P, I], here b = 1
		self.starting_cfvs_as_P1 = results.root_cfvs_both_players[0,P1,:] # 0, because batches = 1
		self.starting_cfvs_as_P2 = results.root_cfvs_both_players[0,P2,:] # 0, because batches = 1





#
