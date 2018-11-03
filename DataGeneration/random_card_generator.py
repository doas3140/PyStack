'''
	Samples random card combinations.
'''
import numpy as np

from Settings.game_settings import game_settings
from Settings.arguments import arguments

class RandomCardGenerator():
	def __init__(self):
		pass

	def generate_cards(self, count):
		''' Samples a random set of cards.
			Each subset of the deck of the correct size is sampled with
			uniform probability.
		@param: count the number of cards to sample
		@return a vector of cards, represented numerically
		'''
		CC = game_settings.card_count
		out = np.random.choice(CC, count, replace=False) # dtype=np.int32
		return out
		# # marking all used cards
		# used_cards = np.zeros([CC], dtype=bool)
		# # counter for generated cards
		# out = np.zeros([count], dtype=arguments.dtype)
		# generated_cards_count = 0
		# while generated_cards_count < count:
		# 	card = np.random.randint(CC)
		# 	if used_cards[card] == 0:
		# 		out[generated_cards_count] = card
		# 		used_cards[card] = 1
		# 		generated_cards_count += 1
		# return out




card_generator = RandomCardGenerator()
