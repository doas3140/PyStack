'''
	Wraps the calls to the final neural net.
'''
import tensorflow as tf

from Settings.arguments import arguments
from Nn.net_builder import nnBuilder

class ValueNn():
	def __init__(self, pretrained_weights=False, verbose=1):
		''' Loads the neural net from disk.
		'''
		# set input and output layer names
		# (must match keras model layer names)
		self.input_layer_name = 'input'
		self.output_layer_name = 'zero_sum_output'
		# set checkpoint and profiler (optional) dir
		self.model_dir = arguments.model_path
		self.profiler_dir = arguments.profiler_path
		# load keras model
		self.keras_model, self.x_shape, self.y_shape = nnBuilder.build_net()
		if verbose > 0:
			print('NN architecture:')
			self.keras_model.summary()
		if pretrained_weights:
			# load model weights
			# self.keras_model = tf.keras.models.load_model(arguments.final_model_path) # doesnt need building (can be faster)
			self.keras_model.load_weights(arguments.final_model_path)


	def get_value(self, inputs, output):
		''' Gives the neural net output for a batch of inputs.
		@param: inputs An (N,I) tensor containing N instances of
				neural net inputs. See @{net_builder} for details of each input.
		@param: output An (N,O) tensor in which to store N sets of
				neural net outputs. See @{net_builder} for details of each output.
		'''
		output[:,:] = self.keras_model.predict(inputs)




#
