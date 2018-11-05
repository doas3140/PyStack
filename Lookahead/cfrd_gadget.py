'''
	Uses the the CFR-D gadget to generate opponent ranges for re-solving.
	See [Solving Imperfect Information Games Using Decomposition]
	(http://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf)
'''

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from tools import tools
from Game.card_tools import card_tools

class CFRDGadget():
	def __init__(self, board, player_range, opponent_cfvs):
		''' Constructor
		@param: board board card
		@param: player_range an initial range vector for the opponent
		@param: opponent_cfvs the opponent counterfactual values vector used for re-solving
		'''
		CC = game_settings.card_count
		assert(board)
		self.input_opponent_range = player_range.copy()
		self.input_opponent_value = opponent_cfvs.copy()
		self.curent_opponent_values = np.zeros([CC], dtype=arguments.dtype)
		self.regret_epsilon = 1.0/100000000
		# 2 stands for 2 actions: play/terminate
		self.opponent_reconstruction_regret = np.zeros([2,CC], dtype=arguments.dtype)
		self.play_current_strategy = np.zeros([CC], dtype=arguments.dtype)
		self.terminate_current_strategy = np.ones([CC], dtype=arguments.dtype)
		# holds achieved CFVs at each iteration so that we can compute regret
		self.total_values = np.zeros([CC], dtype=arguments.dtype)
		self.terminate_regrets = np.zeros([CC], dtype=arguments.dtype)
		self.play_regrets = np.zeros([CC], dtype=arguments.dtype)
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
		play_values = current_opponent_cfvs
		terminate_values = self.input_opponent_value
		# 1.0 compute current regrets
		self.total_values = play_values * self.play_current_strategy
		self.total_values_p2 = self.total_values_p2 or np.zeros_like(self.total_values)
		self.total_values_p2 = terminate_values * self.terminate_current_strategy
		self.total_values += self.total_values_p2
		self.play_current_regret = self.play_current_regret or np.zeros_like(play_values)
		self.terminate_current_regret = self.terminate_current_regret or np.zeros_like(self.play_current_regret)
		self.play_current_regret = play_values.copy()
		self.play_current_regret -= self.total_values
		self.terminate_current_regret = terminate_values.copy()
		self.terminate_current_regret -= self.total_values
		# 1.1 cumulate regrets
		self.play_regrets += self.play_current_regret
		self.terminate_regrets += self.terminate_current_regret
		# 2.0 we use cfr+ in reconstruction
		self.terminate_regrets = np.clip( self.terminate_regrets, self.regret_epsilon, arguments.max_number )
		self.play_regrets = np.clip( self.play_regrets, self.regret_epsilon, arguments.max_number )
		self.play_possitive_regrets = self.play_regrets
		self.terminate_possitive_regrets = self.terminate_regrets
		# 3.0 regret matching
		self.regret_sum = self.regret_sum or np.zeros_like(self.play_possitive_regrets)
		self.regret_sum = self.play_possitive_regrets.copy()
		self.regret_sum += self.terminate_possitive_regrets
		self.play_current_strategy = self.play_possitive_regrets.copy()
		self.terminate_current_strategy = self.terminate_possitive_regrets.copy()
		self.play_current_strategy /= self.regret_sum
		self.terminate_current_strategy /= self.regret_sum
		# 4.0 for poker, the range size is larger than the allowed hands
		# we need to make sure reconstruction does not choose a range
		# that is not allowed
		self.play_current_strategy *= self.range_mask
		self.terminate_current_strategy *= self.range_mask
		self.input_opponent_range = self.input_opponent_range or np.zeros_like(self.play_current_strategy)
		self.input_opponent_range = self.play_current_strategy.copy()
		return self.input_opponent_range




#
