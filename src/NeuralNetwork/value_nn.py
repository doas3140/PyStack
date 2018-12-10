'''
	Wraps the calls to the final neural net.
'''
import os
import tensorflow as tf

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from NeuralNetwork.metrics import BasicHuberLoss, masked_huber_loss

class ValueNn():
	def __init__(self, street, pretrained_weights=False, aux=False, verbose=1):
		# set directories
		street_name = card_to_string.street2name(street)
		self.model_dir_path = os.path.join(arguments.model_path, street_name)
		self.model_path = os.path.join(self.model_dir_path, arguments.final_model_name)
		# load model or create one
		if pretrained_weights:
			self.keras_model = tf.keras.models.load_model( self.model_path,
								   custom_objects = {'loss':BasicHuberLoss(delta=1.0),
								   					 'masked_huber_loss':masked_huber_loss} )
		else: # create keras model
			self.keras_model, self.x_shape, self.y_shape = self.build_net()
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


	def build_net(self):
		''' Builds a neural net with architecture specified by @{arguments.net}.
		@return a newly constructed neural net
		@return input shape (ex: [224,224,3] if img)
		@return output shape (ex: [10] if 10 classes)
		'''
		num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
		num_hands, num_players = constants.hand_count, constants.players_count
		# input and output parameters
		num_output = num_hands * num_players
		num_input = num_output + 1 + num_cards + num_suits + num_ranks
		input_shape = [num_input]
		output_shape = [num_output]
		# neural network architecture
		m_input = tf.keras.layers.Input(input_shape, name='input')
		# slicing off pot size and board ([1, hands x 2 + pot_size + board] -> [1, hands x 2])
		ranges = tf.keras.layers.Lambda(lambda x: x[ : , :num_output ], name='input_ranges')(m_input)
		# reconstruct mask for not possible ranges (where ranges are 0)
		mask = tf.keras.layers.Lambda(lambda x: tf.where( tf.greater(x, tf.constant(0.0, dtype=tf.float32)),
														  tf.ones_like(x), tf.zeros_like(x) ), name='mask')(ranges)
		# feed forward part
		ff = m_input
		for i, num_neurons in enumerate(arguments.num_neurons):
			names = [s.format(i) for s in ('dense_{}', 'relu_{}', 'dropout_{}', 'batch_norm_{}')]
			ff = tf.keras.layers.Dense(num_neurons, name=names[0])(ff)
			ff = tf.keras.layers.Dropout(rate=0.05, name=names[2])(ff)
			ff = tf.keras.layers.BatchNormalization(name=names[3])(ff)
			ff = tf.keras.layers.PReLU(name=names[1])(ff)
		ff = tf.keras.layers.Dense(num_output, name='feed_forward_output')(ff)
		# mask neural network output
		values = tf.keras.layers.multiply([ff,mask], name='masked_output')
		# compute zero-sum output
		estimated_value = tf.keras.layers.dot([values,ranges], axes=1, name='estimated_values')
		estimated_value = tf.keras.layers.Lambda(lambda x: x/2, name='division_by_2')(estimated_value)
		m_output = tf.keras.layers.subtract([values,estimated_value], name='zero_sum_output')
		model = tf.keras.models.Model(m_input, m_output)
		return model, input_shape, output_shape




#
