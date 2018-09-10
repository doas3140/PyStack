'''
	Wraps the calls to the final neural net.
'''

from Settings.arguments import arguments

class ValueNn():
	def __init__(self):
		''' Loads the neural net from disk.
		'''
		pass
		# 0.0 select the correct model cpu/gpu
		# 1.0 load model information
		# import GPU modules only if needed
		# 2.0 load model


	def get_value(self, inputs, output):
		''' Gives the neural net output for a batch of inputs.
		@param: inputs An (N,I) tensor containing N instances of
				neural net inputs. See @{net_builder} for details of each input.
		@param: output An (N,O) tensor in which to store N sets of
				neural net outputs. See @{net_builder} for details of each output.
		'''
		pass




#
