'''
	Implements the same interface as @{value_nn}, but without uses terminal
	equity evaluation instead of a neural net.

	Can be used to replace the neural net during debugging.
'''

from ..Nn.bucketer import Bucketer
from ..TerminalEquity.terminal_equity import TerminalEquity
from ..Settings.game_settings import game_settings
from ..Game.card_tools import card_tools
from ..Settings.arguments import arguments

class MockNnTerminal():
	def __init__(self):
		''' Creates an equity matrix with entries for every possible pair of buckets.
		'''
		pass
		# filling equity matrix


	def get_value(self, inputs, outputs):
		''' Gives the expected showdown equity of the two players' ranges.
		@param: inputs An (N,I) tensor containing N instances of neural net inputs.
				See @{net_builder} for details of each input.
		@param: outputs An (N,O) tensor in which to store N sets of expected showdown
				counterfactual values for each player.
		'''
		pass




#
