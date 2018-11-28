''' Builds the neural net architecture.

	For M buckets, the neural net inputs have size 2*M+1, containing range
	vectors over buckets for each player, as well as a feature capturing the
	pot size. These are arranged as [{p1_range}, {p2_range}, pot_size].

	The neural net outputs have size 2*M, containing counterfactual value
	vectors over buckets for each player. These are arranged as
	[{p1_cfvs}, {p2_cfvs}].
'''

from Nn.bucketer import Bucketer
from Settings.arguments import arguments
from Settings.game_settings import game_settings
import tensorflow as tf

class NetBuilder():
	def __init__(self):
		pass


	def build_net(self):
		''' Builds a neural net with architecture specified by @{arguments.net}.
		@return a newly constructed neural net
		@return input shape (ex: [224,224,3] if img)
		@return output shape (ex: [10] if 10 classes)
		'''
		# input and output parameters
		bucketer = Bucketer()
		bucket_count = bucketer.get_bucket_count()
		player_count = 2
		num_output = bucket_count * player_count
		num_input = num_output + 1
		input_shape = [num_input]
		output_shape = [num_output]
		# neural network architecture
		m_input = tf.keras.layers.Input(input_shape, name='input')
		# slicing off pot size ([1,2001] -> [1,2000])
		sp = tf.keras.layers.Lambda(lambda x: x[ : , :-1 ], name='input_ranges')(m_input)
		# feed forward part
		ff = m_input
		for i in range(arguments.num_layers):
			names = [s.format(i) for s in ('dense_{}', 'prelu_{}')]
			ff = tf.keras.layers.Dense(arguments.num_neurons, name=names[0])(ff)
			ff = tf.keras.layers.PReLU(name=names[1])(ff)
		ff = tf.keras.layers.Dense(num_output, name='feed_forward_output')(ff)
		# dot product of both (feed forward and player ranges)
		d = tf.keras.layers.dot([ff,sp], axes=1, name='dot_product')
		# repeat this number from shape [1] -> [2000]
		d = tf.keras.layers.RepeatVector(num_output, name='repeat_scalar')(d)
		d = tf.keras.layers.Flatten(name='flatten')(d)
		# divide it by 2 and subtract from neural net output
		d = tf.keras.layers.Lambda(lambda x: -x/2, name='divide_by_2')(d)
		m_output = tf.keras.layers.add([ff,d], name='zero_sum_output')
		model = tf.keras.models.Model(m_input, m_output)
		return model, input_shape, output_shape




nnBuilder = NetBuilder()
