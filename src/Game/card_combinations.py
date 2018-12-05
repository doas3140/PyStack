import numpy as np

from Settings.constants import constants


class CardCombinations():
	def __init__(self):
		self.C = {}
		self.max_choose = 55
		self._init_choose()


	def _init_choose(self):
		for i in range(0, self.max_choose+1):
			for j in range(0, self.max_choose+1):
				self.C[i*self.max_choose + j] = 0

		for i in range(0,self.max_choose+1):
			self.C[i*self.max_choose] = 1
			self.C[i*self.max_choose + i] = 1

		for i in range(1,self.max_choose+1):
			for j in range(1,i+1):
				self.C[i*self.max_choose + j] = self.C[(i-1)*self.max_choose + j-1] + self.C[(i-1)*self.max_choose + j]


	def choose(self, n, k):
		return self.C[n*self.max_choose + k]


	def count_possible_boards_with_player_cards(self, street):
		''' counts the number of possible boards if 2 cards where already taken (in players hand)
			the answer will be the same for all player's holding cards
		'''
		num_cards_on_board = constants.board_card_count[street-1]
		max_cards_on_board = constants.board_card_count[-1]
		max_cards_in_deck = constants.card_count
		# counting total card combinations
		num_cards_in_deck = max_cards_in_deck - num_cards_on_board
		num_cards_to_pick = max_cards_on_board - num_cards_on_board
		total_card_combos = self.choose(num_cards_in_deck, num_cards_to_pick)
		# counting possible card combos w/ card1 (from players hand)
		card1 = 1
		num_cards_in_deck = max_cards_in_deck - num_cards_on_board - card1
		num_cards_to_pick = max_cards_on_board - num_cards_on_board - card1
		card1_card_combos = self.choose(num_cards_in_deck, num_cards_to_pick)
		# counting possible card combos w/ card2 (from players hand)
		card2 = 1
		num_cards_in_deck = max_cards_in_deck - num_cards_on_board - card1 - card2 # (w/out card1)
		num_cards_to_pick = max_cards_on_board - num_cards_on_board - card2
		card2_card_combos = self.choose(num_cards_in_deck, num_cards_to_pick)
		# counting possible combos
		num_possible_boards = total_card_combos - card1_card_combos - card2_card_combos
		return num_possible_boards




card_combinations = CardCombinations()
