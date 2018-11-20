'''
	Gives allowed bets during a game.
	Bets are restricted to be from a list of predefined fractions of the pot.
'''
import numpy as np

from Settings.arguments import arguments

class BetSizing():
	def __init__(self, pot_fractions=None):
		''' Constructor
		@param: pot_fractions (num_fractions,) a list of fractions of the pot
				which are allowed as bets, sorted in ascending (min->max) order
		'''
		if pot_fractions is not None: self.pot_fractions = pot_fractions
		else: self.pot_fractions = np.array([1], dtype=arguments.dtype)


	def get_possible_bets(self, node):
		''' Gives the bets which are legal at a game state.
		@param: node a representation of the current game state, with fields:
				* bets (2,): the number of chips currently committed by each player
				 * current_player (): the currently acting player
		@return (N,2) tensor where N is the number of new possible game states,
				containing N sets of new commitment levels for each player
		'''
		current_player = node.current_player
		assert (current_player == 0 or current_player == 1, 'Wrong player for bet size computation')
		opponent = 1 - current_player
		opponent_bet = node.bets[opponent]
		assert(node.bets[current_player] <= opponent_bet)
		# compute min possible raise size
		max_raise_size = arguments.stack - opponent_bet # == call_size
		min_raise_size = opponent_bet - node.bets[current_player]
		min_raise_size = max(min_raise_size, arguments.ante)
		min_raise_size = min(max_raise_size, min_raise_size)
		if min_raise_size == 0:
			return np.zeros([], dtype=arguments.int_dtype) # (N,P), when N = 0
		elif min_raise_size == max_raise_size: # all in
			out = np.full([1,2], opponent_bet, dtype=arguments.int_dtype)
			out[0][current_player] = opponent_bet + min_raise_size
			return out # (N,P)
		else:
			# iterate through all bets and check if they are possible
			if node.num_bets == 0:
				fractions = self.pot_fractions[0]
			elif node.num_bets == 1:
				fractions = self.pot_fractions[1]
			else:
				fractions = self.pot_fractions[2]
			max_possible_bets_count = len(fractions) + 1 # we can always go allin
			out = np.full([max_possible_bets_count,2], opponent_bet, dtype=arguments.int_dtype)
			# take pot size after opponent bet is called
			pot = opponent_bet * 2
			used_bets_count = 0
			# try all pot fractions bet and see if we can use them
			for i in range(len(fractions)):
				raise_size = pot * fractions[i]
				if raise_size >= min_raise_size and raise_size < max_raise_size:
					out[used_bets_count, current_player] = opponent_bet + raise_size
					used_bets_count += 1
			# adding allin
			assert (used_bets_count <= max_possible_bets_count)
			out[used_bets_count, current_player] = opponent_bet + max_raise_size
			used_bets_count += 1
			return out[ :used_bets_count , : ]




#
