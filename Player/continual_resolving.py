'''
	Performs the main steps of continual re-solving, tracking player range
	and opponent counterfactual values so that re-solving can be done at each
	new game state.
'''

from ..Lookahead.resolving import Resolving
from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..Game.card_tools import card_tools

class ContinualResolving():
	def __init__(self):
		''' Does a depth-limited solve of the game's first node.
		'''
		self.starting_player_range = card_tools.get_uniform_range(np.array([]))
    	self.resolve_first_node()


	def resolve_first_node(self):
		''' Solves a depth-limited lookahead from the first node of the game
			to get opponent counterfactual values.
			The cfvs are stored in the field `starting_cfvs_p1`.
			Because this is the first node of the game,
			exact ranges are known for both players,
			so opponent cfvs are not necessary for solving.
		'''
		pass
		# create the starting ranges
		# create re-solving and re-solve the first node
		# store the initial CFVs


	def start_new_hand(self, state):
		''' Re-initializes the continual re-solving to start a new game
			from the root of the game tree.
		@param: state the first state where the re-solving player acts in the new game
				(a table of the type returned by @{protocol_to_node.parse_state})
		'''
		pass


	def _resolve_node(self, node, state):
		''' Re-solves a node to choose the re-solving player's next action.
		@param: node the game node where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parsed_state_to_node})
		@param: state the game state where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parse_state})
		'''
		pass
		# 1.0 first node and P1 position
    	# no need to update an invariant since this is the very first situation
		# the strategy computation for the first decision node has been already set up
		# 2.0 other nodes - we need to update the invariant
		# 2.1 update the invariant based on actions we did not make
		# 2.2 re-solve


	def _update_invariant(self, node, state):
		''' Updates the player's range and the opponent's counterfactual values
			to be consistent with game actions since the last re-solved state.
			Updates it only for actions we did not make,
			since we update the invariant for our action as soon as we make it.
		@param: node the game node where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parsed_state_to_node})
		@param: state the game state where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parse_state})
		'''
		pass
		# 1.0 street has changed
		# 1.1 opponent cfvs
        # if the street has changed, the resonstruction API simply gives us CFVs
		# 1.2 player range
        # if street has change, we have to mask out the colliding hands
		# 2.0 first decision for P2
		# 3.0 handle game within the street


	def compute_action(self, node, state):
		''' Re-solves a node and chooses the re-solving player's next action.
		@param: node the game node where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parsed_state_to_node})
		@param: state the game state where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parse_state})
		@return an action sampled from the re-solved strategy at the given state
				, with the fields:
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips to raise (if `action` is raise)
		'''
		pass


	def _sample_bet(self, node, state):
		''' Samples an action to take from the strategy at the given game state.
		@param: node the game node where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parsed_state_to_node})
		@param: state the game state where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parse_state})
		@return an index representing the action chosen
		'''
		pass
		# 1.0 get the possible bets in the node
		# 2.0 get the strategy for the current hand
		# 3.0 sample the action by doing cumsum and uniform sample
		# 4.0 update the invariants based on our action


	def _bet_to_action(self, node, sampled_bet):
		''' Converts an internal action representation into a cleaner format.
		@param: node the game node where the re-solving player is to act
				(a table of the type returned by @{protocol_to_node.parsed_state_to_node})
		@param: sampled_bet the index of the action to convert
		@return a table specifying the action, with the fields:
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips to raise (if `action` is raise)
		'''
		pass




#
