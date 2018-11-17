'''
	Game constants which define the game played by DeepStack.
'''
from tools import tools

class GameSettings():
	def __init__(self):
		# the number of card suits in the deck
		self.suit_count = 4
		# the number of card ranks in the deck
		self.rank_count = 13
		# the total number of cards in the deck
		self.card_count = self.suit_count * self.rank_count
		# the number of public cards dealt in the game (revealed after the first betting round)
		self.board_card_count = [0, 3, 4, 5]
		self.hand_card_count = 2
		self.hand_count = tools.choose(self.card_count, self.hand_card_count)
		self.limit_bet_sizes = [2, 2, 4, 4]
		self.limit_bet_cap = 4
		self.nl = True



game_settings = GameSettings()
