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
	def __init__(self):
		''' Does a depth-limited solve of the game's first node '''
		board = np.zeros([]) # w/out cards
		self.starting_player_range = card_tools.get_uniform_range(board)
		self.terminal_equity = TerminalEquity()
		self.starting_opponent_cfvs_as_P1 = self._resolve_first_node(constants.players.P1)
		self.starting_opponent_cfvs_as_P2 = self._resolve_first_node(constants.players.P2)


	def start_new_hand(self, card1, card2, player_is_small_blind):
		''' Re-initializes the continual re-solving to start a new game
			from the root of the game tree
		@param: card: string of 2 chars. first is rank (if letter then only capital) and second suit (lower case).
				ex: '2c', '6d', 'Jh', 'Ks', 'Ad'. note: for 10 use 'Ts' not '10s'
		'''
		card1, card2 = card_to_string.string_to_card(card1), card_to_string.string_to_card(card2)
		P1, P2 = constant.players.P1, constant.players.P2
		self.last_node = None
		self.decision_id = 0
		self.player_position = P1 if player_is_small_blind else P2
		self.hand_id = card_tools.get_hole_index([card1, card2])


	def compute_action(self, board_string, player_bet, opponent_bet):
		''' Re-solves a node and chooses the re-solving player's next action '''
		sb, bb, P1, P2 = arguments.sb, arguments.bb, constant.players.P1, constant.players.P2
		# create node
		node = Node()
		node.board = card_to_string.string_to_board(board_string)
		node.street = card_tools.board_to_street(node.board)
		node.current_player = self.player_position
		# note: P1 is always playing small blind, P2 - big blind, but players are sometimes swaped
		P1_bet, P2_bet = (player_bet,opponent_bet) if self.player_position == P1 else (opponent_bet,player_bet)
		node.bets = np.array([P1_bet, P2_bet], dtype=arguments.dtype)
		bets_are_initial = (P1_bet == sb and P2_bet == bb) or (P1_bet == bb and P2_bet == sb)
		node.num_bets = 1 if bets_are_initial else 0
		# resolve and sample bet
		self._resolve_node(node)
		sampled_bet = self._sample_bet()
		# update the invariants based on our action # [I] = [I]
		self.opponent_cfvs_bounded = self.resolving.get_action_cfv(sampled_bet)
		# [I] *= [I]
		self.player_range *= self.resolving.get_action_strategy(sampled_bet)
		self.player_range = card_tools.normalize_range(node.board, self.player_range)
		# update history variables
		self.decision_id += 1
		self.last_bet = sampled_bet
		self.last_node = node
		# return action
		if sampled_bet == constants.actions.fold:
			return {'action':'fold', 'amount': -1}
		elif sampled_bet == constants.actions.ccall:
			return {'action':'call', 'amount': -1}
		else:
			return {'action':'raise', 'amount': sampled_bet}


	def _resolve_first_node(self, opponent_position):
		''' Solves a depth-limited lookahead from the first node of the game
			to get opponent counterfactual values. The cfvs are returned.
			Because this is the first node of the game,
			exact ranges are known for both players,
			so opponent cfvs are not necessary for solving.
		'''
		# create starting node
		first_node = Node()
		first_node.board = np.zeros([])
		first_node.street = 1
		first_node.current_player = opponent_position
		first_node.bets = np.array([arguments.sb, arguments.bb], dtype=arguments.dtype)
		first_node.num_bets = 1
		# set board
		self.terminal_equity.set_board(first_node.board)
		# create the starting ranges
		player_range = card_tools.get_uniform_range(first_node.board)
		opponent_range = card_tools.get_uniform_range(first_node.board)
		# create re-solving and re-solve the first node
		self.first_node_resolving = Resolving(self.terminal_equity)
		self.first_node_resolving.resolve(first_node, player_range, opponent_range=opponent_range)
		# return the initial CFVs
		return self.first_node_resolving.get_root_cfv()


	def _resolve_node(self, node):
		''' Re-solves a node to choose the re-solving player's next action '''
		# 1.0 first node and P1 player_position
		if self.decision_id == 0 and self.player_position == constants.players.P1:
			# no need to update an invariant since this is the very first situation
			# the strategy computation for the first decision node has been already set up
			self.player_range = self.starting_player_range.copy()
			self.resolving = self.first_node_resolving
		else: # 2.0 other nodes - we need to update the invariant
			assert(not node.terminal and node.current_player == self.player_position)
			# 2.1 update the invariant based on actions we did not make
			self._update_player_ranges_and_opponent_cfvs(node)
			# set terminal equity
			self.terminal_equity.set_board(node.board)
			# 2.2 re-solve
			self.resolving = Resolving(self.terminal_equity)
			self.resolving.resolve(node, self.player_range, opponent_cfvs=self.opponent_cfvs_bounded)


	def _update_player_ranges_and_opponent_cfvs(self, node):
		''' Updates the player's range and the opponent's counterfactual values
			to be consistent with game actions since the last re-solved state. '''
		P1, P2 = constants.players.P1, constants.players.P2
		# 1.0 street has changed
		if self.last_node and self.last_node.street != node.street:
			assert(self.last_node.street + 1 == node.street)
			# opponent cfvs: if the street has changed, the resonstruction API simply gives us CFVs
			self.opponent_cfvs_bounded = self.resolving.get_chance_action_cfv(self.last_bet, node.board)
			# player range: if street has change, we have to mask out the colliding hands
			self.player_range = card_tools.normalize_range(node.board, self.player_range)
		elif self.decision_id == 0: # 2.0 first decision for P2
			assert(self.player_position == P2 and node.street == 1)
			self.player_range = self.starting_player_range.copy()
			opponent_position = 1 - self.player_position
			start_cfvs = self.starting_opponent_cfvs_as_P1 if opponent_position == P1 else self.starting_opponent_cfvs_as_P2
			self.opponent_cfvs_bounded = start_cfvs.copy()
		else: # 3.0 handle game within the street
			assert(self.last_node.street == node.street)


	def _sample_bet(self):
		''' Samples an action to take from the strategy at the given game state '''
		# 1.0 get the possible bets in the node
		possible_bets = self.resolving.get_possible_actions() # [len(node.children)]
		# 2.0 get the strategy for the current hand since the strategy is computed for all hands
		hand_strategy = np.zeros([ len(possible_bets) ], dtype=arguments.dtype)
		for bet_amount in possible_bets:
			action_strategy = self.resolving.get_action_strategy(bet_amount) # [I] (for each info set)
			hand_strategy[i] = action_strategy[self.hand_id]
		assert(abs(1 - hand_strategy.sum()) < 0.001)
		# 3.0 sample the action
		sampled_bet = np.random.choice(possible_bets, p=hand_strategy)
		print( "strat: {}, bets: {}, sampled_bet: {}".format(hand_strategy, possible_bets, sampled_bet) )
		return sampled_bet






#
