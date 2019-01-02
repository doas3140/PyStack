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
		@param: [0-5] :vector of board cards (int)
		@param: [I]   :initial opponent counterfactual values vector used for re-solving
		'''
		HC = constants.hand_count
		# init variables for this state
		self.cfvs = np.zeros([2,HC], dtype=arguments.dtype)
		self.regrets = np.zeros([2,HC], dtype=arguments.dtype)
		self.strategy = np.zeros([2,HC], dtype=arguments.dtype)
		# 2 possible actions are: Terminal (fold action), Follow (continue playing)
		self.F, self.T = 0, 1 # first dimension indexes of previously defined variables
		# store initial cfvs (used to get terminal values)
		self.cfvs[self.T] = opponent_cfvs.copy()
		self.strategy[self.T] = np.ones([HC], dtype=arguments.dtype)
		# init range/strategy mask for masking out impossible hands
		self.mask = card_tools.get_possible_hands_mask(board) # [I]


	def compute_opponent_range(self, current_opponent_cfvs):
		''' Uses one iteration of the gadget game to generate an opponent
			range for the current re-solving iteration.
		@param: [I] :vector of cfvs that the opponent receives with the current strategy in the re-solve game
		@return [I] :opponent range vector for this iteration
		'''
		HC = constants.hand_count
		# store cfvs, got from solving sub-game (if we continue playing)
		self.cfvs[self.F] = current_opponent_cfvs
		# conpute total possible values (depends on current strategy)
		# [I] = sum([2,I] * [2,I], axis=0)
		total_cfvs = np.sum(self.cfvs * self.strategy, axis=0)
		# add current regrets (for both actions) to cumulative regrets:
		# broadcasting total_cfvs: [I] -> [2,I]
		# [2,I] = [2,I] - [I]
		self.regrets += self.cfvs - total_cfvs
		# use cfr+
		self.regrets = np.clip(self.regrets, constants.regret_epsilon, constants.max_number)
		# use regret matching to compute strategies for both actions
		# broadcasting sum(regrets): [I] -> [2,I]
		# [2,I] = [2,I] / [I]
		self.strategy = self.regrets / np.sum(self.regrets)
		# compute and add current regrets to cumulative regrets: self.play_regrets and self.terminate_regrets
		# [I] = [I] * [I] + [I] * [I]
		# for poker, the range size is larger than the allowed hands, so we need
		# to make sure reconstruction does not choose a range that is not allowed
		# broadcasting mask: [I] -> [2,I]
		# [2,I] *= [I]
		self.strategy *= self.mask
		# return strategy of Follow (continue playing) action
		# here range = strategy, thats why we can return strategy
		return self.strategy[self.F]




#
