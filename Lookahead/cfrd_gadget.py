'''
	Uses the the CFR-D gadget to generate opponent ranges for re-solving.
	See [Solving Imperfect Information Games Using Decomposition]
	(http://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf)
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..tools import tools
from ..Game.card_tools import card_tools

class CFRDGadget():
	def __init__(self, board, player_range, opponent_cfvs):
		''' Constructor
		@param: board board card
		@param: player_range an initial range vector for the opponent
		@param: opponent_cfvs the opponent counterfactual values vector used for re-solving
		'''
		pass
		# 2 stands for 2 actions: play/terminate
		# holds achieved CFVs at each iteration so that we can compute regret
		# init range mask for masking out impossible hands


	def compute_opponent_range(self, current_opponent_cfvs, iteration):
		''' Uses one iteration of the gadget game to generate an opponent
			range for the current re-solving iteration.
		@param: current_opponent_cfvs the vector of cfvs that the opponent
				receives with the current strategy in the re-solve game
		@param: iteration the current iteration number of re-solving
		@return the opponent range vector for this iteration
		'''
		pass
		# 1.0 compute current regrets
		# 1.1 cumulate regrets
		# 2.0 we use cfr+ in reconstruction
		# 3.0 regret matching
		# 4.0 for poker, the range size is larger than the allowed hands
    	# we need to make sure reconstruction does not choose a range
    	# that is not allowed




#
