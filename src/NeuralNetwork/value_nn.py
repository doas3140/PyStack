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
	def __init__(self, street, pretrained_weights=False, approximate='root_nodes', verbose=1):
		'''
		@param: int  :current street/round
		@param: bool :to load pretrained model or init random weights
		@param: str  :approximate current street "root_nodes"/"leaf_nodes"
		@param: int  :display output if >0
		'''
		# set directories
		self.approximate = approximate # set to approximate leaf or root nodes of specified street
		street_name = card_to_string.street_to_name(street)
		self.model_dir_path = os.path.join(arguments.model_path, street_name)
		model_name = '{}.{}.hdf5'.format(arguments.model_filename, self.approximate)
		self.model_path = os.path.join(self.model_dir_path, model_name)
		# set input, output shapes
		self._set_shapes()
		# load model or create one
		if pretrained_weights:
			self.keras_model = tf.keras.models.load_model( self.model_path,
								   custom_objects = {'loss':BasicHuberLoss(delta=1.0),
													 'masked_huber_loss':masked_huber_loss} )
		else: # create keras model
			self.keras_model = self._build_net()
		# print architecture summary
		if verbose > 0:
			print('NN architecture:')
			self.keras_model.summary()


	def predict(self, inputs, out):
		''' Gives the neural net output for a batch of inputs
		@param: [b,nnI] :tensor containing b batches instances of neural net inputs
		@param: [b,nnO] :tensor in which to store b batches of neural net outputs
		'''
		total_elements, batch_size = inputs.shape[0], 10000
		for i in range(0, total_elements, batch_size):
			start, end = i, i + batch_size
			end = end if end < total_elements else total_elements
			out[ start:end, : ] = self.keras_model.predict_on_batch(inputs[ start:end, : ])


	def _set_shapes(self):
		''' sets self.x_shape and self.y_shape '''
		num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
		num_hands, num_players = constants.hand_count, constants.players_count
		# input and output parameters
		num_output = num_hands * num_players
		num_input = num_output + 1 + num_cards + num_suits + num_ranks
		self.x_shape = [num_input]
		self.y_shape = [num_output]


	def _build_net(self):
		''' creates neural network architecture '''
		num_output = self.y_shape[0]
		m_input = tf.keras.layers.Input(self.x_shape, name='input')
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
			ff = tf.keras.layers.Dropout(rate=0.10, name=names[2])(ff)
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
		return model




#
