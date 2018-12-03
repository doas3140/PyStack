'''
	Wraps the calls to the final neural net.
'''
import os
import tensorflow as tf

from Settings.arguments import arguments
from Nn.net_builder import nnBuilder
from Nn.basic_huber_loss import BasicHuberLoss, masked_huber_loss

class ValueNn():
	def __init__(self, street, pretrained_weights=False, aux=False, verbose=1):
		''' Loads the neural net from disk.
		'''
		# set input and output layer names
		# (must match keras model layer names)
		self.input_layer_name = 'input'
		self.output_layer_name = 'zero_sum_output'
		# set checkpoint and profiler (optional) dir
		street_name = street2name(street)
		self.model_dir_path = os.path.join(arguments.model_path, street_name)
		# load model or create one
		if pretrained_weights:
			self.model_path = os.path.join(self.model_dir_path, arguments.final_model_name)
			self.keras_model = tf.keras.models.load_model( self.model_path,
								   custom_objects = {'loss':BasicHuberLoss(delta=1.0),
								   					 'masked_huber_loss':masked_huber_loss} )
		else:
			# load keras model
			self.keras_model, self.x_shape, self.y_shape = nnBuilder.build_net()
		# print architecture summary
		if verbose > 0:
			print('NN architecture:')
			self.keras_model.summary()


	def get_value(self, inputs, output):
		''' Gives the neural net output for a batch of inputs.
		@param: inputs An (N,I) tensor containing N instances of
				neural net inputs. See @{net_builder} for details of each input.
		@param: output An (N,O) tensor in which to store N sets of
				neural net outputs. See @{net_builder} for details of each output.
		'''
		output[:,:] = self.keras_model.predict(inputs)


def street2name(street):
    if street == 1:
        return 'preflop'
    elif street == 2:
        return 'flop'
    elif street == 3:
        return 'turn'
    elif street == 4:
        return 'river'



#
