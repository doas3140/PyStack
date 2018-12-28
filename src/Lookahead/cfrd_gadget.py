'''
	Uses the the CFR-D gadget to generate opponent ranges for re-solving.
	See [Solving Imperfect Information Games Using Decomposition]
	(http://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf)
'''
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_tools import card_tools

class CFRDGadget():
	def __init__(self, board, opponent_cfvs):
		''' Constructor
		@param: board board card
		@param: opponent_cfvs the opponent counterfactual values vector used for re-solving
		'''
		HC = constants.hand_count
		# store initial cfvs (used to get terminal values)
		self.input_opponent_cfvs = opponent_cfvs.copy()							# [I]
		# update regrets in each iteration
		self.terminate_regrets  = np.zeros([HC], dtype=arguments.dtype)			# [I]
		self.play_regrets       = np.zeros([HC], dtype=arguments.dtype)			# [I]
		# store updated strategy (from regrets)
		self.terminate_strategy = np.ones([HC], dtype=arguments.dtype)			# [I]
		self.play_strategy      = np.zeros([HC], dtype=arguments.dtype)			# [I]
		# init range mask for masking out impossible hands
		self.range_mask = card_tools.get_possible_hands_mask(board) 			# [I]


	def compute_opponent_range(self, current_opponent_cfvs):
		''' Uses one iteration of the gadget game to generate an opponent
			range for the current re-solving iteration.
		@param: current_opponent_cfvs the vector of cfvs that the opponent
				receives with the current strategy in the re-solve game
		@param: iteration the current iteration number of re-solving
		@return the opponent range vector for this iteration
		'''
		HC = constants.hand_count
		# remove first dimension (batches), which should be always be 1
		play_values = current_opponent_cfvs.reshape([HC])
		terminate_values = self.input_opponent_cfvs.reshape([HC])
		# compute and add current regrets to cumulative regrets: self.play_regrets and self.terminate_regrets
		# [I] = [I] * [I] + [I] * [I]
		total_values = (self.play_strategy * play_values) + (self.terminate_strategy * terminate_values)
		# [I] += [I]
		self.terminate_regrets += terminate_values - total_values
		self.play_regrets      += play_values - total_values
		# use cfr+ in reconstruction
		self.terminate_regrets = np.clip( self.terminate_regrets, constants.regret_epsilon, constants.max_number )
		self.play_regrets      = np.clip( self.play_regrets,      constants.regret_epsilon, constants.max_number )
		# regret matching
		# [I] = [I] + [I]
		regret_sum = self.play_regrets + self.terminate_regrets
		# [I] = [I] / [I]
		self.terminate_strategy = self.terminate_regrets / regret_sum
		self.play_strategy      = self.play_regrets      / regret_sum
		# for poker, the range size is larger than the allowed hands
		# we need to make sure reconstruction does not choose a range
		# that is not allowed
		self.terminate_strategy *= self.range_mask
		self.play_strategy      *= self.range_mask
		# return play strategy
		input_opponent_range = self.play_strategy
		return input_opponent_range




#
