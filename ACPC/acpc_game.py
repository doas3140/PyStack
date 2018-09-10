'''
	Handles communication to and from DeepStack using the ACPC protocol.
	For details on the ACPC protocol, see
	<http://www.computerpokercompetition.org/downloads/documents/protocols/protocol.pdf>
'''

from ..ACPC.network_communication import ACPCNetworkCommunication
from ..ACPC.protocol_to_node import ACPCProtocolToNode
from ..Settings.arguments import arguments
from ..Settings.constants import constants

class ACPCGame():
	def __init__(self):
		pass

	def connect(self, server, port):
		''' Connects to a specified ACPC server which acts as the dealer.
		@param: server the server that sends states to DeepStack, which responds
				with actions
		@param: port the port to connect on
		@see network_communication.connect
		'''
		pass


	def get_next_situation(self):
		''' Receives and parses the next poker situation where DeepStack must act.
			Blocks until the server sends a situation where DeepStack acts.
		@return the parsed state representation of the poker situation
				(see @{protocol_to_node.parse_state})
		@return a public tree node for the state
				(see @{protocol_to_node.parsed_state_to_node})
		'''
		pass
		# 1.0 get the message from the dealer
		# 2.0 parse the string to our state representation
		# 3.0 figure out if we should act


	def play_action(self, adviced_action):
		''' Informs the server that DeepStack is playing a specified action.
		@param: adviced_action a table specifying the action chosen by Deepstack,
				with the fields:
				* `action`: an element of @{constants.acpc_actions}
				* `raise_amount`: the number of chips raised (if `action` is raise)
		'''
		pass




#
