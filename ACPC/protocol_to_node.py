'''
	Converts between DeepStack's internal representation and the ACPC protocol
	used to communicate with the dealer.
	For details on the ACPC protocol, see
	<http://www.computerpokercompetition.org/downloads/documents/protocols/protocol.pdf>
'''

from ..Settings.arguments import arguments
from ..Settings.constants import constants
from ..tools import tools
from ..Game.card_to_string_conversion import card_to_string

class ACPCProtocolToNode():
	def __init__(self):
		pass

	def _parse_actions(self, actions):
		''' Parses a list of actions from a string representation.
		@param: actions a string representing a series of actions in ACPC format
		@return a list of actions, each of which is a table with fields:
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips raised (if `action` is raise)
		'''
		pass


	def _parse_state(self, state):
		''' Parses a set of parameters that represent a poker state, from a string
 			representation.
		@param: state a string representation of a poker state in ACPC format
		@return a table of state parameters, containing the fields:
				* `position`: the acting player
				* `hand_id`: a numerical id for the hand
				* `actions`: a list of actions which reached the state, for each
							 betting round - each action is a table with fields:
							 * `action`: an element of @{constants.acpc_actions}
							 * `raise_amount`: the number of chips raised
							 				   (if `action` is raise)
				* `actions_raw`: a string representation of actions for each betting round
				* `board`: a string representation of the board cards
				* `hand_p1`: a string representation of the first player's private hand
				* `hand_p2`: a string representation of the second player's private hand
		'''
		pass


	def _convert_actions_street(self, actions, street, all_actions):
		''' Processes a list of actions for a betting round.
		@param: actions a list of actions (see @{_parse_actions})
		@param: street the betting round on which the actions takes place
		@param: all_actions A list which the actions are appended to.
				Fields `player`, `street`, and `index` are added to each action.
		'''
		pass


	def _convert_actions(self, actions):
		''' Processes all actions.
		@param: actions a list of actions for each betting round
		@return a of list actions, processed with @{_convert_actions_street} and
				concatenated
		'''
		pass


	def _process_parsed_state(self, parsed_state):
		''' Further processes a parsed state into a format understandable by DeepStack.
		@param: parsed_state a parsed state returned by @{_parse_state}
		@return a table of state parameters, with the fields:
				* `position`: which player DeepStack is (element of @{constants.players})
				* `current_street`: the current betting round
				* `actions`: a list of actions which reached the state, for each
							 betting round - each action is a table with fields:
							 * `action`: an element of @{constants.acpc_actions}
							 * `raise_amount`: the number of chips raised (if `action` is raise)
							 * `actions_raw`: a string representation of actions for each betting round
				* `all_actions`: a concatenated list of all of the actions in `actions`,
								 with the following fields added:
								 * `player`: the player who made the action
								 * `street`: the betting round on which the action was taken
								 * `index`: the index of the action in `all_actions`
				* `board`: a string representation of the board cards
				* `hand_id`: a numerical id for the current hand
				* `hand_string`: a string representation of DeepStack's private hand
				* `hand_id`: a numerical representation of DeepStack's private hand
				* `acting_player`: which player is acting (element of @{constants.players})
				* `bet1`, `bet2`: the number of chips committed by each player
		'''
		pass
		# 1.0 figure out the current street
		# 2.0 convert actions to player actions
		# 3.0 current board
		# 5.0 compute bets


	def _compute_bets(self, processed_state):
		''' Computes the number of chips committed by each player at a state.
		@param: processed_state a table containing the fields returned by
				@{_process_parsed_state}, except for `bet1` and `bet2`
		@return the number of chips committed by the first player
		@return the number of chips committed by the second player
		'''
		pass


	def _get_acting_player(self, processed_state):
		''' Gives the acting player at a given state.
		@param: processed_state a table containing the fields returned by
				@{_process_parsed_state}, except for `acting_player`, `bet1`, and `bet2`
		@return the acting player, as defined by @{constants.players}
		'''
		pass
		# has the street changed since the last action?
		# is the hand over?
		# the acting player is the opponent of the one who made the last action


	def parse_state(self, state):
		''' Turns a string representation of a poker state into a
			table understandable by DeepStack.
		@param: state a string representation of a poker state, in ACPC format
		@return a table of state parameters, with the fields:
				* `position`: which player DeepStack is (element of @{constants.players})
				* `current_street`: the current betting round
				* `actions`: a list of actions which reached the state, for each
							 betting round - each action is a table with fields:
							 * `action`: an element of @{constants.acpc_actions}
							 * `raise_amount`: the number of chips raised (if `action` is raise)
				* `actions_raw`: a string representation of actions for each betting round
				* `all_actions`: a concatenated list of all of the actions in `actions`,
								 with the following fields added:
								 * `player`: the player who made the action
								 * `street`: the betting round on which the action was taken
								 * `index`: the index of the action in `all_actions`
				* `board`: a string representation of the board cards
				* `hand_id`: a numerical id for the current hand
				* `hand_string`: a string representation of DeepStack's private hand
				* `hand_id`: a numerical representation of DeepStack's private hand
				* `acting_player`: which player is acting (element of @{constants.players})
				* `bet1`, `bet2`: the number of chips committed by each player
		'''
		pass


	def parsed_state_to_node(self, processed_state):
		''' Gets a representation of the public tree node which corresponds to a
			processed state.
		@param: processed_state a processed state representation returned by
				@{parse_state}
		@return a table representing a public tree node, with the fields:
				* `street`: the current betting round
				* `board`: a (possibly empty) vector of board cards
				* `current_player`: the currently acting player
				* `bets`: a vector of chips committed by each player
		'''
		pass


	def _bet_to_protocol_action(self, adviced_action):
		''' Converts an action taken by DeepStack into a string representation.
		@param: adviced_action the action that DeepStack chooses to take, with fields
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips to raise (if `action` is raise)
		@return a string representation of the action
		'''
		pass


	def action_to_message(self, last_message, adviced_action):
		''' Generates a message to send to the ACPC protocol server,
			given DeepStack's chosen action.
		@param: last_message the last state message sent by the server
		@param: adviced_action the action that DeepStack chooses to take, with fields
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips to raise (if `action` is raise)
		@return a string messsage in ACPC format to send to the server
		'''
		pass




#
