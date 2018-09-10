''' Builds the neural net architecture.
	Uses torch's [nn package](https://github.com/torch/nn/blob/master/README.md).

	For M buckets, the neural net inputs have size 2*M+1, containing range
	vectors over buckets for each player, as well as a feature capturing the
	pot size. These are arranged as [{p1\_range}, {p2\_range}, pot\_size].

	The neural net outputs have size 2*M, containing counterfactual value
	vectors over buckets for each player. These are arranged as
	[{p1\_cfvs}, {p2\_cfvs}].
'''

from ..Nn.bucketer import Bucketer
from ..Settings.arguments import arguments
from ..Settings.game_settings import game_settings

class NetBuilder():
	def __init__(self):
		pass


	def build_net(self):
		''' Builds a neural net with architecture specified by @{arguments.net}.
		@return a newly constructed neural net
		'''
		pass
		# run the lua interpreter on the architecture from the command line
		# to get the list of layers
		# build the network from the layers




nnBuilder = NetBuilder()
