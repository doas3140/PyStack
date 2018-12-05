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
	def __init__(self, board, player_range, opponent_cfvs):
		''' Constructor
		@param: board board card
		@param: player_range an initial range vector for the opponent
		@param: opponent_cfvs the opponent counterfactual values vector used for re-solving
		'''
		HC = constants.hand_count
		self.total_values_p2 = None
		self.play_current_regret = None
		self.terminate_current_regret = None
		self.regret_sum = None
		assert(board is not None)
		self.input_opponent_range = player_range.copy()
		self.input_opponent_value = opponent_cfvs.copy()
		self.curent_opponent_values = np.zeros([HC], dtype=arguments.dtype)
		self.regret_epsilon = 1.0/100000000
		self.play_current_strategy = np.zeros([HC], dtype=arguments.dtype)
		self.terminate_current_strategy = np.ones([HC], dtype=arguments.dtype)
		# holds achieved CFVs at each iteration so that we can compute regret
		self.total_values = np.zeros([HC], dtype=arguments.dtype)
		self.terminate_regrets = np.zeros([HC], dtype=arguments.dtype)
		self.play_regrets = np.zeros([HC], dtype=arguments.dtype)
		# init range mask for masking out impossible hands
		self.range_mask = card_tools.get_possible_hand_indexes(board)


	def compute_opponent_range(self, current_opponent_cfvs, iteration):
		''' Uses one iteration of the gadget game to generate an opponent
			range for the current re-solving iteration.
		@param: current_opponent_cfvs the vector of cfvs that the opponent
				receives with the current strategy in the re-solve game
		@param: iteration the current iteration number of re-solving
		@return the opponent range vector for this iteration
		'''
		HC = constants.hand_count
		play_values = current_opponent_cfvs.reshape([HC])
		terminate_values = self.input_opponent_value.reshape([HC])
		# 1.0 compute current regrets
		# [I] = [I] * [I]
		self.total_values = self.play_current_strategy * play_values
		if self.total_values_p2 is None:
			self.total_values_p2 = np.zeros_like(self.total_values)
		if self.play_current_regret is None:
			self.play_current_regret = np.zeros_like(play_values)
		if self.terminate_current_regret is None:
			self.terminate_current_regret = np.zeros_like(self.play_current_regret)
		# [I] = [I] * [I]
		self.total_values_p2 = terminate_values * self.terminate_current_strategy
		# [I] += [I]
		self.total_values += self.total_values_p2
		# [I] = [I]
		self.play_current_regret = play_values.copy()
		# [I] = [I]
		self.play_current_regret -= self.total_values
		# [I] = [I]
		self.terminate_current_regret = terminate_values.copy()
		# [I] -= [I]
		self.terminate_current_regret -= self.total_values
		# 1.1 cumulate regrets
		# [I] += [I]
		self.play_regrets += self.play_current_regret
		# [I] += [I]
		self.terminate_regrets += self.terminate_current_regret
		# 2.0 we use cfr+ in reconstruction
		self.terminate_regrets = np.clip( self.terminate_regrets, self.regret_epsilon, constants.max_number )
		self.play_regrets = np.clip( self.play_regrets, self.regret_epsilon, constants.max_number )
		# [I] = [I]
		self.play_possitive_regrets = self.play_regrets
		# [I] = [I]
		self.terminate_possitive_regrets = self.terminate_regrets
		# 3.0 regret matching
		if self.regret_sum is None:
			self.regret_sum = np.zeros_like(self.play_possitive_regrets)
		# [I] = [I]
		self.regret_sum = self.play_possitive_regrets.copy()
		# [I] += [I]
		self.regret_sum += self.terminate_possitive_regrets
		# [I] = [I]
		self.play_current_strategy = self.play_possitive_regrets.copy()
		self.terminate_current_strategy = self.terminate_possitive_regrets.copy()
		# [I] /= [I]
		self.play_current_strategy /= self.regret_sum
		self.terminate_current_strategy /= self.regret_sum
		# 4.0 for poker, the range size is larger than the allowed hands
		# we need to make sure reconstruction does not choose a range
		# that is not allowed
		# [I] *= [I]
		self.play_current_strategy *= self.range_mask
		self.terminate_current_strategy *= self.range_mask
		if self.input_opponent_range is None:
			self.input_opponent_range = np.zeros_like(self.play_current_strategy)
		# [I] = [I]
		self.input_opponent_range = self.play_current_strategy.copy()
		return self.input_opponent_range




#
