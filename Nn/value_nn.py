'''
	Wraps the calls to the final neural net.
'''
import tensorflow.keras as keras

from Settings.arguments import arguments

class ValueNn():
	def __init__(self):
		''' Loads the neural net from disk.
		'''
		self.mlp = keras.models.load_model(net_file + '.model')
		print('NN architecture:')
		self.mlp.summary()


	def get_value(self, inputs, output):
		''' Gives the neural net output for a batch of inputs.
		@param: inputs An (N,I) tensor containing N instances of
				neural net inputs. See @{net_builder} for details of each input.
		@param: output An (N,O) tensor in which to store N sets of
				neural net outputs. See @{net_builder} for details of each output.
		'''
		output[:,:] = self.mlp.predict(inputs)




#
