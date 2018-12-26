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
from Player.cache import Cache


class ContinualResolving():
	def __init__(self, verbose=1):
		''' Does a depth-limited solve of the game's first node '''
		HC = constants.hand_count
		self.verbose = verbose
		self.uniform_range = np.full([HC], 1/HC, dtype=arguments.dtype)
		self.terminal_equity = TerminalEquity()
		self.cache = Cache(dir_path=arguments.cache_path)
		self._compute_initial_opponent_cfvs()


	def start_new_hand(self, card1, card2, player_is_small_blind):
		''' Re-initializes the continual re-solving to start a new game
			from the root of the game tree
		@param: card: string of 2 chars. first is rank (if letter then only capital) and second suit (lower case).
				ex: '2c', '6d', 'Jh', 'Ks', 'Ad'. note: for 10 use 'Ts' not '10s'
		'''
		P1, P2 = constants.players.P1, constants.players.P2
		card1, card2 = card_to_string.string_to_card(card1), card_to_string.string_to_card(card2)
		hand = np.sort([card1,card2]) # must be ordered
		self.prev_street, self.prev_action = 1, None
		self.player_position = P1 if player_is_small_blind else P2
		self.holding_hand_idx = card_tools.get_hole_index(hand)
		self.terminal_equity.set_board(np.zeros([])) # set empty board
		# init player range and opponent cfvs
		self.player_range = self.uniform_range.copy()
		start_cfvs = self.starting_cfvs_as_P1 if self.player_position == P2 else self.starting_cfvs_as_P2
		self.opponent_cfvs = start_cfvs.copy()


	def _get_chance_action_cfv(self, node, resolve_results, action_idx):
		''' Gives the average counterfactual values for the opponent during
			re-solving after a chance event
			(the betting round changes and more cards are dealt).
			Used during continual re-solving to track opponent cfvs.
		'''
		if node.street == 1 and self.cache.exists(node.bets):
			next_street_cfvs = self.cache.get_next_street_cfvs(node.bets)
		else: # resolve_results.next_street_cfvs is not None:
			next_street_cfvs = resolve_results.next_street_cfvs
		# save cfvs for particular board
		for i, next_board in enumerate(resolve_results.next_boards):
			if (node.board == next_board).all():
				board_cfvs = next_street_cfvs[:,i,:,:]
		# get next street root node outputs. shape = [self.num_pot_sizes * self.batch_size, P, I]
		# assert(self.num_pot_sizes * self.batch_size == board_cfvs.shape[0])
		# convert action idx to batch index
		action = resolve_results.actions[action_idx]
		batch_index = resolve_results.action_to_index[action] # probably dont need a_idx -> a -> a_idx (still, box_outputs 1st dim = batch x nodes)
		# get cfvs for current player, given some action
		cfvs = board_cfvs[ batch_index , self.player_position ] # ? - turetu but priesingai? 1-p?
		pot = resolve_results.next_round_pot_sizes[batch_index]
		return cfvs * pot


	def compute_action(self, board_string, player_bet, opponent_bet):
		''' Re-solves a node and chooses the re-solving player's next action '''
		node = self._create_node(board_string, player_bet, opponent_bet)
		# if street changed (last node was chance node), then update cfvs and ranges
		if self.prev_street+1 == node.street:
			node.board = card_to_string.string_to_board(board_string)
			# opponent cfvs: if the street has changed, the resonstruction API simply gives us CFVs
			self.opponent_cfvs = self._get_chance_action_cfv(node, self.prev_results, self.prev_action)
			# player range: if street has change, we have to mask out the colliding hands
			mask = self.get_possible_hand_indexes(node.board)
			self.player_range *= mask						# mask available combinations given particular board
			self.player_range /= self.player_range.sum()	# normalize
			# set terminal equity for new board
			self.terminal_equity.set_board(node.board)
		# resolve and sample bet
		results = self._resolve(node)
		# sample bet
		strategy = results.strategy[ : , 0 , self.holding_hand_idx ] # [A,b,I] -> [A], here b = 1
		assert(abs(1 - strategy.sum()) < 0.001)
		print(strategy)
		print(results.actions)
		action_idx = np.random.choice(np.arange(len(strategy)), p=strategy)
		sampled_bet = results.actions[action_idx]
		print( "strat: {}, bets: {}, sampled_bet: {}".format(np.array2string(strategy, suppress_small=True, precision=3), results.actions, sampled_bet) )
		# update the invariants based on our action # [I] = [I]
		self.opponent_cfvs = results.children_cfvs[action_idx,0,:] # [A,b,I], here b = 1
		# [I] *= [I]
		self.player_range *= results.strategy[action_idx,0,:] # [A,b,I], here b = 1
		self.player_range /= self.player_range.sum()	# normalize
		# update history variables
		self.prev_results = results
		self.prev_action = action_idx
		self.prev_street = node.street
		# return action
		if sampled_bet == constants.actions.fold:
			return {'action':'fold', 'amount': -1}
		elif sampled_bet == constants.actions.ccall:
			return {'action':'call', 'amount': -1}
		else:
			return {'action':'raise', 'amount': sampled_bet}


	def _resolve(self, node):
		# check if first street, then maybe use cache
		if node.street == 1 and self.cache.exists(node.bets):
			print('LOADING RESOLVE FROM CACHE')
			results = self.cache.get_resolve_results(node.bets)
		else:
			self.resolving = Resolving(self.terminal_equity)
			player_range = np.expand_dims(self.player_range, axis=0) # add batch dimension (b=1)
			results = self.resolving.resolve(node, player_range, opponent_cfvs=self.opponent_cfvs)
			if node.street == 1:
				self.cache.store_resolve_results(node.bets, results)
		# (for testing)
		# if self.verbose != 0:
		# 	for card1 in range(52):
		# 		for card2 in range(card1+1, 52):
		# 			idx = card_tools.get_hole_index([card1,card2])
		# 			c1, c2 = card_to_string.card_to_string(card1), card_to_string.card_to_string(card2)
		# 			print(c1, c2, np.array2string(results.strategy[:,0,idx], suppress_small=True, precision=2))
		return results


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
		print('created node: bets: {}/{} num_bets: {} street: {} i: {} board: {}'.format(node.bets[0],node.bets[1],
				node.num_bets, node.street, node.current_player, node.board ))
		return node


	def _compute_initial_opponent_cfvs(self):
		''' Solves a depth-limited lookahead from the first node of the game
			to get opponent counterfactual values. The cfvs are returned.
			Because this is the first node of the game,
			exact ranges are known for both players,
			so opponent cfvs are not necessary for solving.
			Save in `starting_cfvs_as_P1` and `starting_cfvs_as_P2`
		'''
		P1, P2 = constants.players.P1, constants.players.P2
		if self.cache.exists(bets=[arguments.sb, arguments.bb]):
			results = self.cache.get_resolve_results(bets=[arguments.sb, arguments.bb])
		else: # solve
			self.player_position = P1 # set it temporary (used in _create_node)
			self.prev_street = -1  # set it to beggining (used in _create_node)
			first_node = self._create_node(board_string='', player_bet=arguments.sb, opponent_bet=arguments.bb)
			# set board
			self.terminal_equity.set_board(first_node.board)
			# create re-solving and re-solve the first node
			resolving = Resolving(self.terminal_equity)
			uniform_range = np.expand_dims(self.uniform_range, axis=0) # add batch dimension (b=1)
			results = resolving.resolve(first_node, player_range=uniform_range, opponent_range=uniform_range)
			# store to cache
			self.cache.store_resolve_results(bets=[arguments.sb, arguments.bb], results=results)
		# store first node results
		self.starting_cfvs_as_P1 = results.root_cfvs_both_players[0,P2,:] # 0, because batches = 1
		self.starting_cfvs_as_P2 = results.root_cfvs_both_players[0,P1,:] # 0, because batches = 1





#
