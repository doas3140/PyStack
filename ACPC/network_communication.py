'''
	Handles network communication for DeepStack.
'''

from ..Settings.arguments import arguments

class ACPCNetworkCommunication():
	def __init__(self):
		pass

	def connect(self, server, port):
		''' Connects over a network socket.
		@param: server the server that sends states to DeepStack, and to which
				DeepStack sends actions
		@param: port the port to connect on
		'''
		pass


	def _handshake(self):
		''' Sends a handshake message to initialize network communication.
		'''
		pass


	def send_line(self, line):
		''' Sends a message to the server.
		@param: line a string to send to the server
		'''
		pass


	def get_line(self):
		''' Waits for a text message from the server. Blocks until a message is
			received.
		@return the message received
		'''
		pass


	def close(self):
		''' Ends the network communication.
		'''
		pass




#
