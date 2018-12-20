
import collections
import random
import numpy as np

from TerminalEquity.terminal_equity import TerminalEquity
from Game.card_to_string_conversion import card_to_string
from Game.card_tools import card_tools
from GUI.client import client as browser


class DoylesGame():
	def __init__(self, bot, logger):
		self.bot = bot
		self.logger = logger
		self.terminal_equity = TerminalEquity() # evaluator
		self.player_hand = ['NO','NO']
		self.bot_hand = ['NO','NO']
		self.stack = 20000
		self.ante = 100
		self.sb = 50
		self.bb = 100
		self.street = 0

	def get_new_shuffled_deck(self):
		ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
		suits = ['c', 'd', 'h', 's']
		deck = collections.deque()
		for s in suits:
			for r in ranks:
				deck.append( '{}{}'.format(r,s) )
		random.shuffle(deck)
		return deck

	def start_round(self, starting_player='player'):
		# shuffle deck + draw hands for each player + log it
		self.deck = self.get_new_shuffled_deck()
		self.player_hand[0] = self.deck.pop()
		self.player_hand[1] = self.deck.pop()
		self.bot_hand[0] = self.deck.pop()
		self.bot_hand[1] = self.deck.pop()
		self.logger.start_round(self.player_hand, self.bot_hand)
		# handle other vars + alert bot for preparation
		self.prev_action = 'no_action'
		self.starting_player = starting_player
		is_sb = True if self.starting_player == 'bot' else False
		self.bot.start_new_hand( card1=self.bot_hand[0],
								 card2=self.bot_hand[1],
								 player_is_small_blind=is_sb )
		self.street = 1
		# handle small and big blinds + log them + change browser
		self.player_chips = self.bot_chips = self.stack
		if self.starting_player == 'player':
			self.bot_chips -= self.bb
			self.player_chips -= self.sb
		else:
			self.bot_chips -= self.sb
			self.player_chips -= self.bb
		self.logger.append_action('player','raise',self.stack-self.player_chips)
		self.logger.append_action('bot','raise',self.stack-self.bot_chips)
		browser.change_chips(self.player_chips, self.bot_chips)
		# handle board + show it to browser (dont show opponents hand)
		self.board = ['NO','NO','NO','NO','NO']
		browser.change_cards(self.board, self.player_hand, bot_cards=['NO','NO'])
		# handle current player + start first players turn
		self.current_player = self.starting_player
		self.next_players_turn()

	def game_over(self, winner):
		# show all cards and show that it is noone's turn
		browser.change_cards(self.board, self.player_hand, self.bot_hand)
		self.current_player = 'game_over'
		# change chips and update browser
		if winner == 'player':
			pot = self.stack - self.bot_chips
			self.player_chips = self.stack + pot
			self.bot_chips = self.stack - pot
		elif winner == 'bot':
			pot = self.stack - self.player_chips
			self.player_chips = self.stack - pot
			self.bot_chips = self.stack + pot
		browser.change_chips(self.player_chips, self.bot_chips)
		# append winner to logs
		self.logger.append_winner(winner, won_pot=pot)
		# notify winner
		browser.notify_winner(winner)


	def set_up_next_street(self):
		print('set_up_next_street')
		# set prev action to None (used for rules)
		self.prev_action = 'no_action'
		self.street += 1
		# if street is > 4, then it is over
		if self.street == 5:
			winner = self.get_last_street_winner()
			self.game_over(winner)
			return None
		# handle current player
		if self.starting_player == 'player':
			self.current_player = 'bot'
		else:
			self.current_player = 'player'
		# handle board cards
		if self.street == 2:
			self.board[0] = self.deck.pop()
			self.board[1] = self.deck.pop()
			self.board[2] = self.deck.pop()
		elif self.street == 3:
			self.board[3] = self.deck.pop()
		elif self.street == 4:
			self.board[4] = self.deck.pop()
		# show board to browser (dont show opponents hand)
		browser.change_cards(self.board, self.player_hand, bot_cards=['NO','NO'])


	def next_players_turn(self):
		print('next_players_turn')
		if self.current_player == 'player':
			# do nothing (wait for response from browser)
			browser.notify_new_turn('player')
			print('PLAYERS TURN')
		elif self.current_player == 'bot':
			browser.notify_new_turn('bot')
			print('BOTS TURN')
			self.bot_action()
		else:
			pass



	def after_action_callback(self, action, amount):
		print('after_action_callback')
		# save action to logger and display new chips on browser
		self.logger.append_action(self.current_player, action, amount)
		browser.change_chips(self.player_chips, self.bot_chips)
		# handle fold
		if action == 'fold':
			winner = 'bot' if self.current_player == 'player' else 'player'
			self.game_over(winner)
			return
		# handle call action
		elif action == 'call':
			if self.prev_action == 'call':
				# function below assigns current player and prev_action
				self.set_up_next_street()
			elif self.prev_action == 'raise':
				self.set_up_next_street()
			elif self.prev_action == 'allin':
				for _ in range(self.street,5):
					self.set_up_next_street()
			else:
				self.current_player = 'bot' if self.current_player == 'player' else 'player'
				self.prev_action = action
		# handle allin action
		elif action == 'allin':
			if self.prev_action == 'allin': # simulate all left streets
				for _ in range(self.street,5):
					self.set_up_next_street()
			else:
				self.current_player = 'bot' if self.current_player == 'player' else 'player'
				self.prev_action = action
		# handle raise action
		elif action == 'raise':
			self.current_player = 'bot' if self.current_player == 'player' else 'player'
			self.prev_action = action
		# go to next player
		self.next_players_turn()


	def player_action(self, action, amount):
		print('player_action')
		if self.current_player != 'player':
			return False, None, None
		if action == 'fold':
			pass
		elif action == 'call':
			self.player_chips = min(self.bot_chips, self.player_chips)
		elif action == 'allin':
			self.player_chips = 0
		elif action == 'raise':
			amount = max(amount, self.ante)
			if amount == self.player_chips:
				action = 'allin'
				self.player_chips = 0
			else:
				self.player_chips = min(self.bot_chips, self.player_chips)
				if amount >= self.player_chips:
					self.player_chips = 0
					action = 'allin'
				else:
					self.player_chips -= amount
		else:
			return False, None, None
		print('PLAYER: {} {}'.format(action, amount))
		# handle what happens after action
		self.after_action_callback(action, amount)
		return True, action, amount


	def get_last_street_winner(self):
		if 'NO' in self.board:
			# replace with draw card
			assert(False)
		board = card_to_string.string_to_board( ''.join(self.board) )
		player_hand = card_to_string.string_to_board( ''.join(self.player_hand) )
		player_hand_idx = card_tools.get_hole_index(np.sort(player_hand))
		bot_hand = card_to_string.string_to_board( ''.join(self.bot_hand) )
		bot_hand_idx = card_tools.get_hole_index(np.sort(bot_hand))
		self.terminal_equity.set_board(board)
		strengths = self.terminal_equity.get_hand_strengths()
		player_hand_strength = strengths[ player_hand_idx ]
		bot_hand_strength = strengths[ bot_hand_idx ]
		print('==========================================')
		print('player:', player_hand_strength)
		print('bot:', bot_hand_strength)
		return 'player' if player_hand_strength > bot_hand_strength else 'bot'


	def bot_action(self):
		print('bot_action')
		board = [card for card in self.board if card != 'NO']
		bot_bet = self.stack - self.bot_chips
		player_bet = self.stack - self.player_chips
		res = self.bot.compute_action( board_string=''.join(board),
									   player_bet=bot_bet,
									   opponent_bet=player_bet )
		action, amount = res['action'], res['amount']
		if action == 'fold':
			pass
		elif action == 'call':
			self.bot_chips = min(self.bot_chips, self.player_chips)
		elif action == 'allin':
			self.bot_chips = 0
		elif action == 'raise':
			amount = max(amount, self.ante)
			if amount == self.bot_chips:
				action = 'allin'
				self.bot_chips = 0
			else:
				self.bot_chips = min(self.bot_chips, self.player_chips)
				if amount >= self.bot_chips:
					self.bot_chips = 0
					action = 'allin'
				else:
					self.bot_chips -= amount
		else:
			return False
		# handle what happens after action
		self.after_action_callback(action, amount)
