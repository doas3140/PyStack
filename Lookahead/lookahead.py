
'''
	A depth-limited lookahead of the game tree used for re-solving.
'''

from ..Lookahead.lookahead_builder import LookaheadBuilder
from ..TerminalEquity.terminal_equity import TerminalEquity
from ..Lookahead.cfrd_gadget import CFRDGadget
from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Settings.game_settings import game_settings
from ..tools import tools

class Lookahead():
	def __init__(self):
		self.builder = LookaheadBuilder(self)


	def build_lookahead(self, tree):
		''' Constructs the lookahead from a game's public tree.
			Must be called to initialize the lookahead.
		@param: tree a public tree
		'''
		pass


	def resolve_first_node(self, player_range, opponent_range):
		''' Re-solves the lookahead using input ranges.
			Uses the input range for the opponent instead of a gadget range,
			so only appropriate for re-solving the root node of the game tree
			(where ranges are fixed).
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_range a range vector for the opponent
		'''
		pass


	def resolve(self, player_range, opponent_cfvs):
		''' Re-solves the lookahead using an input range for the player and
			the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
		@{build_lookahead} must be called first.
		@param: player_range a range vector for the re-solving player
		@param: opponent_cfvs a vector of cfvs achieved by the opponent
				before re-solving
		'''
		pass


	def _compute(self):
		''' Re-solves the lookahead.
		'''
		pass
		# 1.0 main loop
		# 2.0 at the end normalize average strategy
		# 2.1 normalize root's CFVs


	def _compute_current_strategies(self):
		''' Uses regret matching to generate the players' current strategies.
		'''
		pass


	def _compute_ranges(self):
		''' Using the players' current strategies, computes their
			probabilities of reaching each state of the lookahead.
		'''
		pass
		# 1.0 set regret of empty actions to 0
		# 1.1  regret matching
        # note that the regrets as well as the CFVs have switched player indexing


	def _compute_update_average_strategies(self, iter):
		''' Updates the players' average strategies with their current strategies.
		@param: iter the current iteration number of re-solving
		'''
		pass
		# copy the ranges of inner nodes and transpose
		# multiply the ranges of the acting player by his strategy


	def _compute_terminal_equities_terminal_equity(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at each lookahead state which is a terminal state of the game.
		'''
		pass
		# call term eq evaluation
		# on river, any call is terminal
		# correctly set the folded player by mutliplying by -1


	def _compute_terminal_equities_next_street_box(self):
		''' Using the players' reach probabilities, calls the neural net to
			compute the players' counterfactual values at the depth-limited
			states of the lookahead.
		'''
		pass
		# now the neural net accepts the input for P1 and P2 respectively,
		# so we need to swap the ranges if necessary
		# now the neural net outputs for P1 and P2 respectively,
		# so we need to swap the output values if necessary


	def get_chance_action_cfv(self, action_index, board):
		''' Gives the average counterfactual values for the opponent during
			re-solving after a chance event
			(the betting round changes and more cards are dealt).
			Used during continual re-solving to track opponent cfvs.
			The lookahead must first be re-solved with
			@{resolve} or @{resolve_first_node}.
		@param: action_index the action taken by the re-solving player
				at the start of the lookahead
		@param: board a tensor of board cards, updated by the chance event
		@return a vector of cfvs
		'''
		pass
		# check if we should not use the first layer for transition call
		# remove fold


	def _compute_terminal_equities(self):
		''' Using the players' reach probabilities, computes their counterfactual
			values at all terminal states of the lookahead.
			These include terminal states of the game and depth-limited states.
		'''
		pass
		# multiply by pot scale factor

	def _compute_cfvs(self):
		''' Using the players' reach probabilities and terminal counterfactual
			values, computes their cfvs at all states of the lookahead.
		'''
		pass
		# player indexing is swapped for cfvs
		# use a swap placeholder to change
		# [[1,2,3], [4,5,6]] into [[1,2], [3,4], [5,6]]


	def _compute_cumulate_average_cfvs(self, iter):
		''' Updates the players' average counterfactual values with their
			cfvs from the current iteration.
		@param: iter the current iteration number of re-solving
		'''
		pass


	def _compute_normalize_average_strategies(self):
		''' Normalizes the players' average strategies.
			Used at the end of re-solving so that we can track
			un-normalized average strategies, which are simpler to compute.
		'''
		pass
		# using regrets_sum as a placeholder container
		# if the strategy is 'empty' (zero reach), strategy does not matter
		# but we need to make sure it sums to one -> now we set to always fold



	def _compute_normalize_average_cfvs(self):
		''' Normalizes the players' average counterfactual values.
			Used at the end of re-solving so that we can track
			un-normalized average cfvs, which are simpler to compute.
		'''
		pass


	def _compute_regrets(self):
		''' Using the players' counterfactual values, updates their
			total regrets for every state in the lookahead.
		'''
		# (CFR+)

	def get_results(self):
		''' Gets the results of re-solving the lookahead.
			The lookahead must first be re-solved with @{resolve} or @{resolve_first_node}.
		@return a table containing the fields:
				* `strategy`: an (A,K) tensor containing the re-solve player's
				strategy at the root of the lookahead, where
				A is the number of actions and K is the range size
				* `achieved_cfvs`: a vector of the opponent's
				average counterfactual values at the root of the lookahead
				* `children_cfvs`: an (A,K) tensor of opponent
				average counterfactual values after each action
				that the re-solve player can take at the root of the lookahead
		'''
		pass
		# 1.0 average strategy
    	# lookahead already computes the averate strategy we just convert the dimensions
		# 2.0 achieved opponent's CFVs at the starting node
		# 3.0 CFVs for the acting player only when resolving first node
		# swap cfvs indexing
		# 4.0 children CFVs
		# IMPORTANT divide average CFVs by average strategy in here


	def _set_opponent_starting_range(self, iteration):
		''' Generates the opponent's range for the current re-solve iteration
			using the @{cfrd_gadget|CFRDGadget}.
		@param: iteration the current iteration number of re-solving
		'''
		pass
		# note that CFVs indexing is swapped, thus the CFVs
		# for the reconstruction player are for player '1'




#
